#include <qpOASES.hpp>
#include "wbc_v2.h"
#include "costom_utils.hpp"
#include "useful_math.h"

namespace CostomAlgorithm {

constexpr double BigDouble = 1e10;

WBC_priority::WBC_priority(
        const double timestep_input, 
        const int model_nv) {
    timestep_ = timestep_input;
    model_nv_ = model_nv;
    initTasks(model_nv_);
    // 足端力矩上下界,物体系下
    tau_upp_stand_L_ << 15, 30, 40;
    tau_low_stand_L_ << -15, -30, -40;
    tau_upp_walk_L_ << 15, 40, 40; 
    tau_low_walk_L_ << -15, -40, -40;

    delta_q_final_kin_ = Eigen::VectorXd::Zero(model_nv_);
    dq_final_kin_ = Eigen::VectorXd::Zero(model_nv_);
    ddq_final_kin_ = Eigen::VectorXd::Zero(model_nv_);
    ddp_final_dyn_ = Eigen::VectorXd::Zero(model_nv_);
    tau_final_dyn_ = Eigen::VectorXd::Zero(model_nv_ - 6);
    Fr_final_dyn_ = Eigen::VectorXd::Zero(12);
}

void WBC_priority::dataBusRead(const DataBus &robotState) {
    robot_state_ptr_ = &robotState;
}

void WBC_priority::dataBusRead(const DataBus *robotState_ptr) {
    robot_state_ptr_ = robotState_ptr;
}

void WBC_priority::setMode(const WBCMode& mode_input) {
    mode_ = mode_input;
    initTasks(model_nv_);   
}

WBCMode WBC_priority::getMode() const {
    return mode_;
}
void WBC_priority::computeDdq(Pin_KinDyn &pinKinDynIn) {

    const int model_nv = robot_state_ptr_->model_nv;
    Eigen::MatrixXd J_contact, dJ_contact, J_swing, dJ_swing;
    Eigen::Vector3d fe_pos_sw_W;
    Eigen::MatrixXd fe_rot_sw_W;
    // 这些量应该拿到DataBus里算好了
    if (robot_state_ptr_->legState == DataBus::LSt) {
        J_contact =  robot_state_ptr_->J_l;
        dJ_contact =  robot_state_ptr_->dJ_l;
        J_swing = robot_state_ptr_->J_r;
        dJ_swing = robot_state_ptr_->dJ_r;
        fe_pos_sw_W = robot_state_ptr_->fe_r_pos_W;
        fe_rot_sw_W = robot_state_ptr_->fe_r_rot_W;
    } else {
        J_contact =  robot_state_ptr_->J_r;
        dJ_contact =  robot_state_ptr_->dJ_r;
        J_swing = robot_state_ptr_->J_l;
        dJ_swing = robot_state_ptr_->dJ_l;
        fe_pos_sw_W = robot_state_ptr_->fe_l_pos_W;
        fe_rot_sw_W = robot_state_ptr_->fe_l_rot_W;
    }
    // task definition
    /// -------- walk -------------
    {
        int id = -1;
        if (mode_ == WBCMode::HardContactMode) {
            id = kin_tasks_walk_.getId("static_Contact");
            kin_tasks_walk_.taskLib[id].J = J_contact;
            kin_tasks_walk_.taskLib[id].dJ = dJ_contact;
        }

        // 任务：冗余关节尽量不动，冗余关节指的是头部（2个），腰（3个）
        // 所以期望值就是0
        id = kin_tasks_walk_.getId("RedundantJoints");
        kin_tasks_walk_.taskLib[id].errX = 
            Eigen::Vector<double, 5>::Zero() - robot_state_ptr_->q.block(21, 0, 5, 1);
        kin_tasks_walk_.taskLib[id].derrX = 
            Eigen::Vector<double, 5>::Zero() - robot_state_ptr_->dq.block(21, 0, 5, 1);

        id = kin_tasks_walk_.getId("Roll_Pitch_Yaw_Pz");
        kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(4);
        Eigen::Matrix3d desRot = 
            eul2Rot(robot_state_ptr_->base_rpy_des(0), 
                    robot_state_ptr_->base_rpy_des(1), 
                    robot_state_ptr_->base_rpy_des(2));
        kin_tasks_walk_.taskLib[id].errX.block<3, 1>(0, 0) = 
            diffRot(robot_state_ptr_->base_rot, desRot);
        kin_tasks_walk_.taskLib[id].errX(3) = 
            robot_state_ptr_->base_pos_des(2) - robot_state_ptr_->q(2);
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(4);
        kin_tasks_walk_.taskLib[id].derrX.block<3, 1>(0, 0) = 
            -robot_state_ptr_-> dq.block<3, 1>(3, 0);
        kin_tasks_walk_.taskLib[id].derrX(3) = 0 - robot_state_ptr_->dq(2);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(4);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(4);
        Eigen::MatrixXd taskMap = Eigen::MatrixXd::Zero(4, 6);
        taskMap(0, 3) = 1;
        taskMap(1, 4) = 1;
        taskMap(2, 5) = 1;
        taskMap(3, 2) = 1;
        // robot_state_ptr_->J_base = [ones(6, 6), zeros(6, 31)]
        // J选出第3到6个元素
        kin_tasks_walk_.taskLib[id].J = taskMap * robot_state_ptr_->J_base;
        kin_tasks_walk_.taskLib[id].dJ = taskMap * robot_state_ptr_->dJ_base;


        id = kin_tasks_walk_.getId("PxPy");
        kin_tasks_walk_.taskLib[id].errX.resize(2);
        kin_tasks_walk_.taskLib[id].errX = 
            robot_state_ptr_->base_vel_des.block<2, 1>(0, 0) * timestep_;
        kin_tasks_walk_.taskLib[id].derrX.resize(2);
        kin_tasks_walk_.taskLib[id].derrX = 
            robot_state_ptr_->dq.block<2, 1>(0, 0) - 
            robot_state_ptr_->base_vel_des.block<2, 1>(0, 0);
        kin_tasks_walk_.taskLib[id].dxDes.resize(2);
        kin_tasks_walk_.taskLib[id].dxDes = robot_state_ptr_->base_vel_des.block<2, 1>(0, 0);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
        taskMap = Eigen::MatrixXd::Zero(2, 6);
        taskMap(0, 0) = 1;
        taskMap(1, 1) = 1;
        // 就是选出dq的前两行 J = [ones(2, 2), zeros(2, 35)]
        kin_tasks_walk_.taskLib[id].J = taskMap * robot_state_ptr_->J_base;
        kin_tasks_walk_.taskLib[id].dJ = taskMap * robot_state_ptr_->dJ_base;

        id = kin_tasks_walk_.getId("PosRot");
        kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].errX.block(0, 0, 3, 1) = 
            robot_state_ptr_->base_pos_des - 
            robot_state_ptr_->q.block(0, 0, 3, 1);
        double pos_x_error_threshold;
        double pos_y_error_threshold;
        double pos_z_error_threshold;
        if (mode_ == WBCMode::HardContactMode) {
            pos_x_error_threshold = 0.02;
            pos_y_error_threshold = 0.02;
            pos_z_error_threshold = 0.005;
        } else if (mode_ == WBCMode::SoftContactMode) {
            pos_x_error_threshold = 0.03;
            pos_y_error_threshold = 0.03;
            pos_z_error_threshold = 0.01;
        }
        if (fabs(kin_tasks_walk_.taskLib[id].errX(0)) >= pos_x_error_threshold)
            kin_tasks_walk_.taskLib[id].errX(0) = 
                pos_x_error_threshold * sign(kin_tasks_walk_.taskLib[id].errX(0));
        if (fabs(kin_tasks_walk_.taskLib[id].errX(1)) >= pos_y_error_threshold)
            kin_tasks_walk_.taskLib[id].errX(1) = 
                pos_y_error_threshold * sign(kin_tasks_walk_.taskLib[id].errX(1));
        if (kin_tasks_walk_.taskLib[id].errX(2) > pos_z_error_threshold){
            kin_tasks_walk_.taskLib[id].errX(2) = pos_z_error_threshold;
        }
        //  求出来是一个so3
        desRot = eul2Rot(robot_state_ptr_->base_rpy_des(0), 
                        robot_state_ptr_->base_rpy_des(1), 
                        robot_state_ptr_->base_rpy_des(2));
        kin_tasks_walk_.taskLib[id].errX.block<3, 1>(3, 0) = 
            diffRot(robot_state_ptr_->base_rot, desRot);
        kin_tasks_walk_.taskLib[id].errX(4) -= 0.05 * robot_state_ptr_->dq(4);
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].dxDes << 
            robot_state_ptr_->base_omega_des, robot_state_ptr_->base_vel_des ;
        kin_tasks_walk_.taskLib[id].J = robot_state_ptr_->J_base;
        kin_tasks_walk_.taskLib[id].dJ = robot_state_ptr_->dJ_base;
        if (mode_ == WBCMode::SoftContactMode) {
            kin_tasks_walk_.taskLib[id].N = 
                Eigen::MatrixXd::Identity(J_contact.cols(), J_contact.cols()) -
                pseudoInv_right_weighted(J_contact, kin_tasks_walk_.taskLib[id].W) *
                    J_contact;
        }

        id = kin_tasks_walk_.getId("SwingLeg");
        if (mode_ == WBCMode::HardContactMode) {
            kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].errX.block<3, 1>(0, 0) = 
                robot_state_ptr_->swing_fe_pos_des_W - fe_pos_sw_W;
            desRot = eul2Rot(
                robot_state_ptr_->swing_fe_rpy_des_W(0),
                robot_state_ptr_->swing_fe_rpy_des_W(1),
                robot_state_ptr_->swing_fe_rpy_des_W(2));
            kin_tasks_walk_.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(fe_rot_sw_W, desRot);      
            kin_tasks_walk_.taskLib[id].errX(4) *= 2;
            kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
        } else if (mode_ == WBCMode::SoftContactMode) {
            kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].errX.block<3, 1>(0, 0) = 
                robot_state_ptr_->swing_fe_pos_des_W - fe_pos_sw_W;
            desRot = eul2Rot(
                robot_state_ptr_->swing_fe_rpy_des_W(0),
                robot_state_ptr_->swing_fe_rpy_des_W(1),
                robot_state_ptr_->swing_fe_rpy_des_W(2));
            kin_tasks_walk_.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(fe_rot_sw_W, desRot);      
            kin_tasks_walk_.taskLib[id].errX(4) *= 2;
            kin_tasks_walk_.taskLib[id].errX *= 1.5;
            kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
        }
        kin_tasks_walk_.taskLib[id].J = J_swing;
        kin_tasks_walk_.taskLib[id].J.block(0, 22, 6, 3).setZero(); // exculde waist joints
        kin_tasks_walk_.taskLib[id].dJ = dJ_swing;
        kin_tasks_walk_.taskLib[id].dJ.block(0, 22, 6, 3).setZero(); // exculde waist joints

        // task 6: hand track
        double l_hip_pitch = robot_state_ptr_->q(28) - robot_state_ptr_->q(34);
        double r_hip_pitch = robot_state_ptr_->q(34) - robot_state_ptr_->q(28);
        Eigen::VectorXd target_arm_q;
        target_arm_q.resize(14);
        target_arm_q << 0.475 - 0.75*r_hip_pitch, -1.12, 1.9, 0.86, -0.356, 0, 0, -0.475 + 0.75*l_hip_pitch, -1.12, -1.9, 0.86, 0.356, 0, 0;

        id = kin_tasks_walk_.getId("HandTrackJoints");
        kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(14);
        kin_tasks_walk_.taskLib[id].errX = target_arm_q - robot_state_ptr_->q.block<14, 1>(7, 0);
    }

    /// -------- stand -------------
    {
        int id = -1;
        if (mode_ == WBCMode::HardContactMode) {
            int id = kin_tasks_stand_.getId("static_Contact");
            kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(12, model_nv);
            Eigen::MatrixXd taskCtMap = Eigen::MatrixXd::Zero(3, 3);
            taskCtMap(0, 0) = 0;
            taskCtMap(1, 1) = 1;
            taskCtMap(2, 2) = 1;
            taskCtMap = 
                robot_state_ptr_->fe_l_rot_W * taskCtMap * 
                robot_state_ptr_->fe_l_rot_W.transpose(); // disable ankle roll joint
            kin_tasks_stand_.taskLib[id].J.block(0, 0, 6, model_nv) =
                robot_state_ptr_->J_l;
            kin_tasks_stand_.taskLib[id].J.block(6, 0, 6, model_nv) =
                robot_state_ptr_->J_r;
            kin_tasks_stand_.taskLib[id].J.block(3, 0, 3, model_nv) = taskCtMap * kin_tasks_stand_.taskLib[id].J.block(3, 0, 3, model_nv);
            kin_tasks_stand_.taskLib[id].J.block(9, 0, 3, model_nv) = taskCtMap * kin_tasks_stand_.taskLib[id].J.block(9, 0, 3, model_nv);
            kin_tasks_stand_.taskLib[id].J.block(0, 22, 12, 3).setZero(); // exculde waist joints
        }

        id = kin_tasks_stand_.getId("HipRPY");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(3);
        Eigen::Matrix3d desRot = eul2Rot(0, 0, 0);
        kin_tasks_stand_.taskLib[id].errX.block<3, 1>(0, 0) = 
            diffRot(robot_state_ptr_->hip_link_rot, desRot);
        Eigen::MatrixXd taskMapRPY = Eigen::MatrixXd::Zero(3, 6);
        taskMapRPY(0, 3) = 1;
        taskMapRPY(1, 4) = 1;
        taskMapRPY(2, 5) = 1;
        kin_tasks_stand_.taskLib[id].J = 
            taskMapRPY * robot_state_ptr_->J_hip_link;
        kin_tasks_stand_.taskLib[id].J.block(0, 22, 3, 3).setZero();
        kin_tasks_stand_.taskLib[id].J.block(0, 6, 3, 14).setZero();


        id = kin_tasks_stand_.getId("Pz");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(1);
        kin_tasks_stand_.taskLib[id].errX(0) = robot_state_ptr_->base_pos_des(2) - robot_state_ptr_->q(2);
        Eigen::MatrixXd taskMap = Eigen::MatrixXd::Zero(1, 6);
        taskMap(0, 2) = 1;
        kin_tasks_stand_.taskLib[id].J = taskMap * robot_state_ptr_->J_base;
        kin_tasks_stand_.taskLib[id].J.block(0, 22, 1, 3).setZero();
        kin_tasks_stand_.taskLib[id].dJ = 
            taskMap * robot_state_ptr_->dJ_base;
        kin_tasks_stand_.taskLib[id].dJ.block(0, 22, 1, 3).setZero();


        id = kin_tasks_stand_.getId("CoMTrack");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].errX = 
           robot_state_ptr_->q.block(0, 0, 2, 1) - 
           robot_state_ptr_->pCoM_W.block(0, 0, 2, 1);
        kin_tasks_stand_.taskLib[id].J = 
            robot_state_ptr_->Jcom_W.block(0, 0, 2, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(0, 6, 2, 14).setZero();

        id = kin_tasks_stand_.getId("CoMXY_HipRPY");
        taskMapRPY = Eigen::MatrixXd::Zero(3, 6);
        taskMapRPY(0, 3) = 1;
        taskMapRPY(1, 4) = 1;
        taskMapRPY(2, 5) = 1;
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(5);
        kin_tasks_stand_.taskLib[id].errX.block(0, 0, 2, 1) = 
            robot_state_ptr_->pCoM_W.head(2) - robot_state_ptr_->q.head(2);
		desRot = eul2Rot(robot_state_ptr_->base_rpy_des(0),
                        robot_state_ptr_->base_rpy_des(1), 
                        robot_state_ptr_->base_rpy_des(2));
        kin_tasks_stand_.taskLib[id].errX.tail(3) = 
            diffRot(robot_state_ptr_->hip_link_rot, desRot);
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(5);
        kin_tasks_stand_.taskLib[id].derrX.head(2) =
            -(robot_state_ptr_->Jcom_W * robot_state_ptr_->dq).head(2);
        kin_tasks_stand_.taskLib[id].derrX.tail(3) =
            -((taskMapRPY * robot_state_ptr_->J_hip_link) * robot_state_ptr_->dq);
        kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(5, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(0, 0, 2, model_nv) = 
            robot_state_ptr_->Jcom_W.block(0, 0, 2, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(2, 0, 3, model_nv) = 
            taskMapRPY * robot_state_ptr_->J_hip_link;
        kin_tasks_stand_.taskLib[id].J.block(2, 22, 3, 3).setZero(); // exculde waist joints
        kin_tasks_stand_.taskLib[id].J.block(2, 6, 3, 14).setZero(); // exculde arm joints
        // std::cout << std::endl << "==========CoMXY_HipRPY==========" << std::endl;
        // CostomUtils::print_vector(kin_tasks_stand_.taskLib[id].errX, 
        //                           "CoMXY_HipRPY: errX");
        // CostomUtils::print_vector(kin_tasks_stand_.taskLib[id].derrX, 
        //                             "CoMXY_HipRPY: errX");

        // define swing arm motion
        Eigen::VectorXd target_arm_q;
        target_arm_q.resize(14);
        target_arm_q << 0.475, -1.12, 1.9, 0.86, -0.356, 0, 0, -0.475, -1.12, -1.9, 0.86, 0.356, 0, 0;

        id = kin_tasks_stand_.getId("HandTrackJoints");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(14);
        kin_tasks_stand_.taskLib[id].errX = target_arm_q - robot_state_ptr_->q.block<14, 1>(7, 0);
 

        // Enter here functions to send actuator commands, like:
        // arm-l: 0-6, arm-r: 7-13, head: 14,15, waist: 16-18, leg-l: 19-24, leg-r: 25-30

        id = kin_tasks_stand_.getId("HeadRP");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].errX(0) = 0 - robot_state_ptr_->q(21);
        kin_tasks_stand_.taskLib[id].errX(1) = 
            robot_state_ptr_->rpy[0] - robot_state_ptr_->q(22);

        id = kin_tasks_stand_.getId("Roll_Pitch_Yaw");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(3);
        desRot = eul2Rot(robot_state_ptr_->base_rpy_des(0), robot_state_ptr_->base_rpy_des(1), robot_state_ptr_->base_rpy_des(2));
        kin_tasks_stand_.taskLib[id].errX = diffRot(robot_state_ptr_->base_rot, desRot);
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].derrX = 
            -robot_state_ptr_->dq.block<3, 1>(3, 0);
        taskMap = Eigen::MatrixXd::Zero(3, 6);
        taskMap(0, 3) = 1;
        taskMap(1, 4) = 1;
        taskMap(2, 5) = 1;
        kin_tasks_stand_.taskLib[id].J = taskMap * robot_state_ptr_->J_base;
        kin_tasks_stand_.taskLib[id].dJ = taskMap * robot_state_ptr_->dJ_base;

        id = kin_tasks_stand_.getId("fixedWaist");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].errX(0) = 0 - robot_state_ptr_->q(23);
        kin_tasks_stand_.taskLib[id].errX(1) = 0 - robot_state_ptr_->q(24);
        kin_tasks_stand_.taskLib[id].errX(2) = 0 - robot_state_ptr_->q(25);
    }
    DataBus::MotionState motionStateCur = robot_state_ptr_->motionState;
    Eigen::MatrixXd dyn_M_inv = robot_state_ptr_->dyn_M_inv;
    Eigen::VectorXd dq = robot_state_ptr_->dq;
    if (motionStateCur == DataBus::Walk || motionStateCur == DataBus::Walk2Stand)
    {
        kin_tasks_walk_.computeAll(dyn_M_inv, dq);
        delta_q_final_kin_ = kin_tasks_walk_.out_delta_q;
        dq_final_kin_ = kin_tasks_walk_.out_dq;
        ddq_final_kin_ = kin_tasks_walk_.out_ddq;
    }
    else if (motionStateCur == DataBus::Stand)
    {
        kin_tasks_stand_.computeAll(dyn_M_inv, dq);
        delta_q_final_kin_ = kin_tasks_stand_.out_delta_q;
        dq_final_kin_ = kin_tasks_stand_.out_dq;
        ddq_final_kin_ = kin_tasks_stand_.out_ddq;
    }
    else
    {
        delta_q_final_kin_ = Eigen::VectorXd::Zero(model_nv);
        dq_final_kin_ = Eigen::VectorXd::Zero(model_nv);
        ddq_final_kin_ = Eigen::VectorXd::Zero(model_nv);
    }

    // print_kin_tasks_walk();
    // print_kin_tasks_standce();

    // final WBC output collection
}

void WBC_priority::computeTau() {
    if (mode_ == WBCMode::HardContactMode) {
        // computeTauHardContact();
        computeTauSoftContact();
    } else if (mode_ == WBCMode::SoftContactMode) {
        computeTauSoftContact();
    }
}

void WBC_priority::computeTauHardContact() {
    // 18 means the sum of dims of delta_r and delta_Fr
    const int variable_dim = 18;
    const int constraint_dim = 22;
    // 等式约束 在qpoases中等式约束的表达形式为 b <= A <= b
    // 将浮动基动力学，加速度松弛，反作用力松弛统一表达为一个等式约束
    /// 具体形式可参考笔记
    // 浮动基选择矩阵，浮动基加速度指令选择矩阵
    Eigen::MatrixXd selector_float_mat, Selector_delatf_Mat, J_contact; 
    selector_float_mat.setZero(6, robot_state_ptr_->model_nv);
    selector_float_mat.block(0, 0, 6, 6).setIdentity();
    Selector_delatf_Mat.setZero(robot_state_ptr_->model_nv, 6);
    Selector_delatf_Mat.block(0, 0, 6, 6).setIdentity();
    J_contact.resize(12, robot_state_ptr_->model_nv);
    J_contact.block(0, 0, 6, robot_state_ptr_->model_nv) =
        robot_state_ptr_->J_l;
    J_contact.block(6, 0, 6, robot_state_ptr_->model_nv) =
        robot_state_ptr_->J_r;

    Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(6, variable_dim); 
    A1.block<6, 6>(0, 0) = 
        selector_float_mat * robot_state_ptr_->dyn_M * Selector_delatf_Mat;
    A1.block<6, 12>(0, 6) = -selector_float_mat * J_contact.transpose();

    Eigen::Vector<double, 6> ubA1 = Eigen::VectorXd::Zero(6);
    Eigen::Vector<double, 6> lbA1 = Eigen::VectorXd::Zero(6);
    ubA1 = 
        selector_float_mat * J_contact.transpose() * robot_state_ptr_->Fr_ff -
        selector_float_mat * robot_state_ptr_->dyn_M * ddq_final_kin_ - selector_float_mat * robot_state_ptr_->dyn_Non ;
    lbA1 = ubA1;

    // 足端摩擦锥约束与力矩约束 单只脚: 4个摩擦锥 + fz限制 + mx,my,mz限制
    Eigen::Matrix3d Rfe;
    if (robot_state_ptr_->legState == DataBus::LSt) {
        Rfe = robot_state_ptr_->fe_l_rot_W;
    }
    else {
        Rfe = robot_state_ptr_->fe_r_rot_W;
    }

    Eigen::Matrix<double, 12, 12> extended_R_w2b;
    extended_R_w2b.setZero();
    extended_R_w2b.block(0, 0, 3, 3) = Rfe.transpose();
    extended_R_w2b.block(3, 3, 3, 3) = Rfe.transpose();
    extended_R_w2b.block(6, 6, 3, 3) = Rfe.transpose();
    extended_R_w2b.block(9, 9, 3, 3) = Rfe.transpose();

    Eigen::MatrixXd A2_block = Eigen::MatrixXd::Zero(16, 12);
    A2_block.block(0, 0, 4, 3) <<
        -1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
        1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
        0.0, -1.0, -1.0 / sqrt(2.0) * miu_,
        0.0, 1.0, -1.0 / sqrt(2.0) * miu_;
    A2_block.block<4, 4>(4, 2) = Eigen::MatrixXd::Identity(4, 4);
    A2_block.block<8, 6>(8, 6) = A2_block.block<8, 6>(0, 0);
    A2_block = A2_block * extended_R_w2b;

    Eigen::VectorXd f_low = Eigen::VectorXd::Zero(16);
    Eigen::VectorXd f_upp = Eigen::VectorXd::Zero(16);
    Eigen::Vector3d tau_upp_fe, tau_low_fe;
    if (robot_state_ptr_->motionState == DataBus::Stand) {
        tau_upp_fe = tau_upp_stand_L_;
        tau_low_fe = tau_low_stand_L_;
    } else {
        tau_upp_fe = tau_upp_walk_L_;
        tau_low_fe = tau_low_walk_L_;
    }

    // 左脚站立,右脚的反作用力上下界保持初始化的0不变
    // 修改左脚相应的值
    std::vector<int> blocks_to_update;//需要更新的起始索引数组
    if (robot_state_ptr_->legState == DataBus::LSt) {
        blocks_to_update = {0};
    } else if (robot_state_ptr_->legState == DataBus::RSt) {
        blocks_to_update = {8};
    } else {
        blocks_to_update = {0, 8};
    }
    
    for (int start : blocks_to_update) {
        f_upp.block(start, 0, 8, 1) << 
            Eigen::Vector4d::Zero(), fr_z_max_, tau_upp_fe;
        f_low.block(start, 0, 8, 1) << 
            Eigen::Vector4d::Constant(-BigDouble), fr_z_min_, tau_low_fe;
    }

    Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(16, 18);
    A2.block<16, 12>(0, 6) = A2_block;
    Eigen::VectorXd lbA2 = Eigen::VectorXd::Zero(16);
    Eigen::VectorXd ubA2 = Eigen::VectorXd::Zero(16);

    lbA2 = f_low - A2_block * robot_state_ptr_->Fr_ff;
    ubA2 = f_upp - A2_block * robot_state_ptr_->Fr_ff;

    // 将等式约束与不等式约束拼接成总的约束
    Eigen::MatrixXd A_total = 
        Eigen::MatrixXd::Zero(constraint_dim, variable_dim);
    A_total.block<6, 18>(0, 0) = A1;
    A_total.block<16, 18>(6, 0) = A2;

    Eigen::VectorXd lbA_total = Eigen::VectorXd::Zero(constraint_dim);
    Eigen::VectorXd ubA_total = Eigen::VectorXd::Zero(constraint_dim);

    lbA_total.block<6, 1>(0, 0) = lbA1;
    lbA_total.block<16, 1>(6, 0) = lbA2;
    ubA_total.block<6, 1>(0, 0) = ubA1;
    ubA_total.block<16, 1>(6, 0) = ubA2;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(variable_dim, variable_dim);
    Eigen::MatrixXd Q2 = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd Q1 = Eigen::MatrixXd::Identity(12, 12);
    if (robot_state_ptr_->motionState == DataBus::Stand){
        H.block<6, 6>(0, 0) = Q2 * 2.0 * 1e7;
        H.block<12, 12>(6, 6) = Q1 * 2.0 * 1e1;
        H(9,9) *= 100;
        H(10,10) *= 100;
        H(15,15) *= 100;
        H(16,16) *= 100;
    }
    else{
        H.block<6, 6>(0, 0) = Q2 * 2.0 * 1e7;
        H.block<12, 12>(6, 6) = Q1 * 2.0 * 1e1;
    }

    // obj: (1/2)x'Hx+x'g
    // s.t. lbA<=Ax<=ubA
    //       lb<=x<=ub
    qpOASES::real_t *H_qpoases = 
        new qpOASES::real_t[variable_dim * variable_dim];
    qpOASES::real_t *A_qpoases = 
        new qpOASES::real_t [constraint_dim * variable_dim];
    qpOASES::real_t *g_qpoases = 
        new qpOASES::real_t[variable_dim];
    qpOASES::real_t *lbA_qpoases = 
        new qpOASES::real_t[constraint_dim];
    qpOASES::real_t *ubA_qpoases = 
        new qpOASES::real_t[constraint_dim];

    copy_Eigen_to_real_t(H_qpoases, H);
    copy_Eigen_to_real_t(A_qpoases, A_total);
    copy_Eigen_to_real_t(lbA_qpoases, lbA_total);
    copy_Eigen_to_real_t(ubA_qpoases, ubA_total);
    for (int i = 0; i < variable_dim; i++) {
        g_qpoases[i] = 0;
    }
    nWSR_ = 200;
    cpu_time_ = timestep_;

    qpOASES::QProblem QP_prob(H.rows(), A_total.rows());
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_LOW;
    QP_prob.setOptions(options);
    qp_status_ = QP_prob.init(
        H_qpoases, g_qpoases, A_qpoases, NULL, NULL, 
        lbA_qpoases, ubA_qpoases, nWSR_, &cpu_time_);
    //    if (res==qpOASES::SUCCESSFUL_RETURN)
    //        printf("WBC-QP: successful_return\n");
    //    else if (res==qpOASES::RET_MAX_NWSR_REACHED)
    //        printf("WBC-QP: max_nwsr\n");
    //    else if (res==qpOASES::RET_INIT_FAILED)
    //        printf("WBC-QP: init_failed\n");

    ddp_final_dyn_ = ddq_final_kin_;
    Fr_final_dyn_ = robot_state_ptr_->Fr_ff;
    if (qp_status_ == qpOASES::SUCCESSFUL_RETURN) {
        qpOASES::real_t xOpt[variable_dim];
        QP_prob.getPrimalSolution(xOpt);
        optimal_var_.resize(variable_dim);
        for (int i = 0; i < variable_dim; i++) {
            optimal_var_(i) = xOpt[i];
        }
        optimal_var_pre_ = optimal_var_;
        ddp_final_dyn_.block<6, 1>(0, 0) += optimal_var_.block<6, 1>(0, 0);
        Fr_final_dyn_ += optimal_var_.block<12, 1>(6, 0);
    } else {
        optimal_var_ = optimal_var_pre_;
    }

    Eigen::VectorXd tauRes;
    tauRes = robot_state_ptr_->dyn_M * ddp_final_dyn_ + 
            robot_state_ptr_->dyn_Non - J_contact.transpose() * Fr_final_dyn_;

    tau_final_dyn_ = tauRes.block(6, 0, robot_state_ptr_->model_nv - 6, 1);

    delete[] H_qpoases;
    delete[] A_qpoases;
    delete[] g_qpoases;
    delete[] lbA_qpoases;
    delete[] ubA_qpoases;
}

void WBC_priority::computeTauSoftContact() {
    // 优化变量[delta_Fr, ddxc delta_ddq,] 
    // MPC输出的反作用力松弛(12维), 接触点质心加速度(12维), ddq的松弛(维度可变)
    // ddq可以自己定义修改,只要修改var_ddq_dim的数值和选择矩阵S_ddq
    // 暂定为质心6 + 双腿12 = 18 
    // const int var_delta_ddq_dim = robot_state_ptr_->model_nv;
    const int var_delta_ddq_dim = 18;
    const int var_all_dim = 12 + 12 + var_delta_ddq_dim;
    Eigen::MatrixXd selector_ddq_mat;
    // selector_ddq_mat.setIdentity(robot_state_ptr_->model_nv, var_delta_ddq_dim);
    selector_ddq_mat.setZero(robot_state_ptr_->model_nv, var_delta_ddq_dim);
    selector_ddq_mat.topLeftCorner(6, 6).setIdentity();
    selector_ddq_mat.bottomRightCorner(12, 12).setIdentity();


    // 等式约束
    // 浮动基动力学约束
    Eigen::MatrixXd selector_float_mat, J_contact;
    selector_float_mat.setZero(6, robot_state_ptr_->model_nv);
    selector_float_mat.block(0, 0, 6, 6).setIdentity();
    J_contact.resize(12, robot_state_ptr_->model_nv);
    J_contact << robot_state_ptr_->J_l, robot_state_ptr_->J_r;

    Eigen::MatrixXd A1(6, var_all_dim);
    A1 << - selector_float_mat * J_contact.transpose(), 
            Eigen::Matrix<double, 6, 12>::Zero(),
            selector_float_mat * robot_state_ptr_->dyn_M * selector_ddq_mat;
    Eigen::VectorXd lbA1(6), ubA1(6);
    lbA1 = selector_float_mat * J_contact.transpose() * robot_state_ptr_->Fr_ff -
           selector_float_mat * robot_state_ptr_->dyn_M * ddq_final_kin_ -
           selector_float_mat * robot_state_ptr_->dyn_Non;
    ubA1 = lbA1;

    // 接触动力学约束 xc_ddot = J * q_ddot + dJ * q
    // 通过有限差分计算dJ
    if (is_first_run_) {
        J_contact_pre_ = J_contact;
        is_first_run_ = false;
    } 
    Eigen::MatrixXd dJ_contact = (J_contact - J_contact_pre_) / timestep_;
    J_contact_pre_ = J_contact;

    Eigen::MatrixXd A2(12, var_all_dim);
    A2 << Eigen::Matrix<double, 12, 12>::Zero(),
          Eigen::Matrix<double, 12, 12>::Identity(),
          - J_contact * selector_ddq_mat;
    Eigen::VectorXd lbA2, ubA2;
    lbA2 = J_contact * ddq_final_kin_ + dJ_contact * dq_final_kin_;
    ubA2 = lbA2;

    // 摩擦锥约束 + mx, my, mz约束
    // 单只脚
    Eigen::MatrixXd A3_block = Eigen::MatrixXd::Zero(20, 12);
    A3_block.block(0, 0 , 4, 3) <<
        -1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
        1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
        0.0, -1.0, -1.0 / sqrt(2.0) * miu_,
        0.0, 1.0, -1.0 / sqrt(2.0) * miu_;
    constexpr double factor = 0.6; // 这个系数的计算可以参考笔记
    A3_block.block(4, 0, 2, 6) <<
        0.0, 0.0, -factor * miu_, 0.0, 0.0, 1.0,
        0.0, 0.0, -factor * miu_, 0.0, 0.0, -1.0;

    constexpr double feet_half_x_length = 0.1225;
    constexpr double feet_half_y_length = 0.04;
    constexpr double factor2 = 0.8;
    A3_block.block(6, 0, 4, 6) <<
        0.0, 0.0, -factor2 * feet_half_x_length, 0.0, 1.0, 0.0,
        0.0, 0.0, -factor2 * feet_half_x_length, 0.0, -1.0, 0.0,
        0.0, 0.0, -factor2 * feet_half_y_length, 1.0, 0.0, 0.0,
        0.0, 0.0, -factor2 * feet_half_y_length, -1.0, 0.0, 0.0;
    A3_block.bottomRightCorner(10, 6) = A3_block.topLeftCorner(10, 6);
    // 转移到物体坐标系
    Eigen::Matrix3d R_fe;
    if (robot_state_ptr_->legState == DataBus::LegState::LSt) {
        R_fe = robot_state_ptr_->fe_l_rot_W;
    } else {
        R_fe = robot_state_ptr_->fe_r_rot_W;
    }
    Eigen::MatrixXd R_extended;
    R_extended.setZero(12 , 12);
    for (int i = 0; i < 4; ++i) {
        R_extended.block(3 * i, 3 * i, 3, 3) = R_fe;
    }
    A3_block = A3_block * R_extended;

    // 拼装成双足完整约束
    Eigen::MatrixXd A3;
    A3.setZero(20, var_all_dim);
    A3.topLeftCorner(20, 12) = A3_block;
    Eigen::VectorXd lbA3, ubA3;
    ubA3 = - A3_block * robot_state_ptr_->Fr_ff;
    lbA3.setConstant(A3_block.rows(), -BigDouble);

    // Fz方向的松弛限制
    Eigen::MatrixXd A4_block, A4;
    A4_block.setZero(2, 12);
    A4_block.block(0, 0, 1, 3) = 
        Eigen::Matrix<double, 1, 3>{0.0, 0.0, 1} * R_fe;
    A4_block.block(1, 6, 1, 3) = 
        Eigen::Matrix<double, 1, 3>{0.0, 0.0, 1} * R_fe;
    A4.setZero(2, var_all_dim);
    A4.topLeftCorner(2, 12) = A4_block;
    // 根据相位计算左右脚Fz,max
    double fr_z_left_min, fr_z_left_max, fr_z_right_min, fr_z_right_max;  
    if (robot_state_ptr_->legState == DataBus::LSt) {
        fr_z_left_min = fr_z_min_;
        fr_z_left_max = fr_z_max_;
        fr_z_right_min = fr_z_min_;
        fr_z_right_max = 20.0;
    } else if (robot_state_ptr_->legState == DataBus::RSt) {
        fr_z_left_min = fr_z_min_;
        fr_z_left_max = 20.0;
        fr_z_right_min = fr_z_min_;
        fr_z_right_max = fr_z_max_;
    } else if (robot_state_ptr_->legState == DataBus::DSt) {
        fr_z_left_min = fr_z_min_;
        fr_z_left_max = fr_z_max_;
        fr_z_right_min = fr_z_min_;
        fr_z_right_max = fr_z_max_;
    }
    Eigen::Vector2d lbA4, ubA4;
    lbA4 = Eigen::Vector2d{fr_z_left_min, fr_z_right_min} - 
            A4_block * robot_state_ptr_->Fr_ff;
    ubA4 = Eigen::Vector2d{fr_z_left_max, fr_z_right_max} -
            A4_block * robot_state_ptr_->Fr_ff;
    // 拼成整个约束
    const int constraint_dim = 
        A1.rows() + A2.rows() + A3.rows() + A4.rows();
    Eigen::MatrixXd A_total(constraint_dim, var_all_dim);
    Eigen::VectorXd lbA_total(constraint_dim), 
                    ubA_total(constraint_dim);
    A_total << A1, A2, A3, A4;
    lbA_total << lbA1, lbA2, lbA3, lbA4;
    ubA_total << ubA1, ubA2, ubA3, ubA4;

    // 目标函数
    Eigen::MatrixXd H_delta_fr, H_ddxc, H_delta_ddq;
    H_delta_fr.setIdentity(12, 12);
    // H_delta_fr.diagonal() <<
    //     10.0, 1.0, 2, 3.0, 3.0, 3.0,
    //     10.0, 1.0, 2, 3.0, 3.0, 3.0;
    H_delta_fr.diagonal() *= 50.0;    
    H_ddxc = Eigen::MatrixXd::Identity(12, 12) * 20.0;
    H_ddxc.diagonal().head(3).setConstant(40.0);
    H_ddxc.diagonal().segment(6, 3).setConstant(40.0);
    H_delta_ddq = 
        Eigen::MatrixXd::Identity(var_delta_ddq_dim ,var_delta_ddq_dim) * 1e6;
    H_delta_ddq.diagonal().segment(var_delta_ddq_dim - 6, 3).setConstant(1e10);
    H_delta_ddq.diagonal().segment(var_delta_ddq_dim - 12, 3).setConstant(1e10);
    // H_delta_ddq.topLeftCorner(6, 6).diagonal().setConstant(1e8);
    // 根据站立状态设计权重值
    if (robot_state_ptr_->legState == DataBus::LSt) {
        // 左脚站立：左脚的反作用力的惩罚很小，左脚接触点加速度惩罚很大
        H_delta_fr.diagonal().head(6).setConstant(10.0);
        H_ddxc.diagonal().tail(6).setConstant(10.0);
    } else if (robot_state_ptr_->legState == DataBus::RSt) {
        H_delta_fr.diagonal().tail(6).setConstant(10.0);
        H_ddxc.diagonal().head(6).setConstant(10.0);
    }
    // 拼成一个大矩阵
    Eigen::MatrixXd H_total;
    H_total.setZero(var_all_dim, var_all_dim);
    H_total.topLeftCorner(12, 12) = 2 * H_delta_fr;
    H_total.block(12, 12, 12, 12) = 2 * H_ddxc;
    H_total.bottomRightCorner(var_delta_ddq_dim, var_delta_ddq_dim) = 
        2 * H_delta_ddq;

    // 求解二次规划
    qpOASES::real_t *H_qpoases = 
        new qpOASES::real_t[var_all_dim * var_all_dim];
    qpOASES::real_t *A_qpoases = 
        new qpOASES::real_t [A_total.rows() * var_all_dim];
    qpOASES::real_t *g_qpoases = 
        new qpOASES::real_t[var_all_dim];
    qpOASES::real_t *lbA_qpoases = 
        new qpOASES::real_t[var_all_dim];
    qpOASES::real_t *ubA_qpoases = 
        new qpOASES::real_t[var_all_dim];

    copy_Eigen_to_real_t(H_qpoases, H_total);
    copy_Eigen_to_real_t(A_qpoases, A_total);
    copy_Eigen_to_real_t(lbA_qpoases, lbA_total);
    copy_Eigen_to_real_t(ubA_qpoases, ubA_total);
    for (int i = 0; i < var_all_dim; ++i) {
        g_qpoases[i] = 0;
    }
    nWSR_ = 200;
    cpu_time_ = timestep_;

    qpOASES::QProblem QP_prob(H_total.rows(), A_total.rows());
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_LOW;
    QP_prob.setOptions(options);
    qp_status_ = QP_prob.init(
        H_qpoases, g_qpoases, A_qpoases, NULL, NULL, 
        lbA_qpoases, ubA_qpoases, nWSR_, &cpu_time_);
    
    ddp_final_dyn_ = ddq_final_kin_;
    Fr_final_dyn_ = robot_state_ptr_->Fr_ff;
    if (qp_status_ == qpOASES::SUCCESSFUL_RETURN) {
        qpOASES::real_t xOpt[var_all_dim];
        QP_prob.getPrimalSolution(xOpt);
        optimal_var_.resize(var_all_dim);
        for (int i = 0; i < var_all_dim; i++) {
            optimal_var_(i) = xOpt[i];
        }
        optimal_var_pre_ = optimal_var_;
        ddp_final_dyn_ += 
            selector_ddq_mat * optimal_var_.tail(var_delta_ddq_dim);
        Fr_final_dyn_ += optimal_var_.head(12);
    } else {
        optimal_var_ = optimal_var_pre_;
    }

    Eigen::VectorXd tauRes;
    tauRes = robot_state_ptr_->dyn_M * ddp_final_dyn_ + 
            robot_state_ptr_->dyn_Non - J_contact.transpose() * Fr_final_dyn_;

    tau_final_dyn_ = tauRes.segment(6, robot_state_ptr_->model_nv - 6);

    // 打印输出debug
    std::cout << std::endl << std::endl << 
    "================= QP PROBLEM DEBUG INFO =================" << std::endl;

    // // 使用匿名函数打印所有矩阵和向量
    // CostomUtils::print_matrix(A1, "浮动基质心动力学约束 A1");
    // CostomUtils::print_vector(lbA1, "lbA1");
    // CostomUtils::print_vector(ubA1, "ubA1");

    // CostomUtils::print_matrix(A2, "接触加速度约束 A2");
    // CostomUtils::print_vector(lbA2, "lbA2");
    // CostomUtils::print_vector(ubA2, "ubA2");

    // std::cout << std::endl << "----- 旋转矩阵 Rfe -----" << std::endl;
    // std::cout << R_fe.format(matrix_clear_fmt) << std::endl;

    // CostomUtils::print_matrix(A3, "摩擦力与力矩约束 A3");
    // CostomUtils::print_vector(lbA3, "lbA3");
    // CostomUtils::print_vector(ubA3, "ubA3");

    // CostomUtils::print_matrix(A4, "摩擦力与力矩约束 A4");
    // CostomUtils::print_vector(lbA4, "lbA4");
    // CostomUtils::print_vector(ubA4, "ubA4");

    // CostomUtils::print_matrix(H_total, "代价矩阵 H");

    // // 打印求解状态
    // std::cout << std::endl << "----- Solver Status -----" << std::endl;
    // std::cout << "QP solve status: " << qp_status_ << " (0=SUCCESS)" << std::endl;
    // std::cout << "CPU time: " << cpu_time_ << " sec" << std::endl;
    // std::cout << "Iterations: " << nWSR_ << std::endl;

    // 打印关键中间变量
    // std::cout << std::endl << "----- Intermediate Variables -----" << std::endl;
    // CostomUtils::print_matrix(J_contact, "J_contact");
    // CostomUtils::print_matrix(dJ_contact, "dJ_contact");
    // CostomUtils::print_vector(ddq_final_kin_, "ddq_final_kin_");
    // CostomUtils::print_vector(dq_final_kin_, "dq_final_kin_");
    // CostomUtils::print_vector(J_contact * ddq_final_kin_, "J_contact * ddq_final_kin_");
    // CostomUtils::print_vector(dJ_contact * dq_final_kin_, "dJ_contact * dq_final_kin_");
    // std::cout << "Current leg state: " << 
    //     static_cast<int>(robot_state_ptr_->legState) << 
    //     " (0=LSt,1=RSt,2=DSt)" << std::endl;

    // // 打印优化结果
    // std::cout << std::endl << "----- Optimization Results -----" << std::endl;
    // CostomUtils::print_vector(optimal_var_.head(12), "地面反作用力松弛");
    // CostomUtils::print_vector(optimal_var_.segment(12, 12), "接触加速度");
    // CostomUtils::print_vector(optimal_var_.tail(18), "质心与关节加速度松弛");
    // CostomUtils::print_vector(tau_final_dyn_.head(6), "Tau result (first 6)");
    // CostomUtils::print_vector(Fr_final_dyn_.head(6), "Fr_final_dyn_ (left)");
    // CostomUtils::print_vector(Fr_final_dyn_.tail(6), "Fr_final_dyn_ (right)");


    // std::cout << std::endl << "================= DEBUG INFO END =================" 
    // << std::endl << std::endl;

    delete[] H_qpoases;
    delete[] A_qpoases;
    delete[] g_qpoases;
    delete[] lbA_qpoases;
    delete[] ubA_qpoases;
}

void WBC_priority::dataBusWrite(DataBus &robotState) {
    robotState.wbc_delta_q_final = delta_q_final_kin_;
    robotState.wbc_dq_final = dq_final_kin_;
    robotState.wbc_ddq_final = ddp_final_dyn_;
    robotState.wbc_tauJointRes = tau_final_dyn_;
    robotState.wbc_FrRes = Fr_final_dyn_;
    robotState.qp_cpuTime = cpu_time_;
    robotState.qp_nWSR = nWSR_;
    robotState.qp_status = qp_status_;
}

void WBC_priority::initTasks(const int model_nv){
    // 初始化任务列表,并将不需要动态修改的部分赋值
    // 1.初始化任务列表
    //  WBC task defined and order build
    ///------------ walk --------------
    kin_tasks_walk_.addTask("static_Contact");
    kin_tasks_walk_.addTask("Roll_Pitch_Yaw_Pz");
    kin_tasks_walk_.addTask("RedundantJoints");
    kin_tasks_walk_.addTask("PxPy");
    kin_tasks_walk_.addTask("SwingLeg");
    kin_tasks_walk_.addTask("HandTrackJoints");
    kin_tasks_walk_.addTask("PosRot");

    std::vector<std::string> taskOrder_walk;
    if (mode_ == WBCMode::HardContactMode) {
        taskOrder_walk.emplace_back("static_Contact");
    }
    taskOrder_walk.emplace_back("PosRot");
    taskOrder_walk.emplace_back("SwingLeg");
    taskOrder_walk.emplace_back("RedundantJoints");
    taskOrder_walk.emplace_back("HandTrackJoints");

    kin_tasks_walk_.buildPriority(taskOrder_walk);

    ///-------- stand ------------
    kin_tasks_stand_.addTask("static_Contact");
    kin_tasks_stand_.addTask("CoMTrack");
    kin_tasks_stand_.addTask("HandTrackJoints");
    kin_tasks_stand_.addTask("HipRPY");
    kin_tasks_stand_.addTask("HeadRP");
    kin_tasks_stand_.addTask("Pz");
    kin_tasks_stand_.addTask("CoMXY_HipRPY");
    kin_tasks_stand_.addTask("Roll_Pitch_Yaw");
    kin_tasks_stand_.addTask("fixedWaist");

    std::vector<std::string> taskOrder_stand;
    if (mode_ == WBCMode::HardContactMode) {
        taskOrder_stand.emplace_back("static_Contact");
    } 
    taskOrder_stand.emplace_back("CoMXY_HipRPY");
    taskOrder_stand.emplace_back("Pz");
    taskOrder_stand.emplace_back("HandTrackJoints");
    taskOrder_stand.emplace_back("HeadRP");

    kin_tasks_stand_.buildPriority(taskOrder_stand);

    // 2.静态部分赋值
    // ------------walk-------------
    {   
        int id = -1;
        if (mode_ == WBCMode::HardContactMode) {
            id = kin_tasks_walk_.getId("static_Contact");           kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
            kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 0;
            kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 0;
            kin_tasks_walk_.taskLib[id].W.diagonal() = 
                Eigen::VectorXd::Ones(model_nv);
        }

        id = kin_tasks_walk_.getId("RedundantJoints");
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(5);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(5);
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(5, 5) * 100;
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(5, 5) * 20;
        // 操作空间就是关节空间
        kin_tasks_walk_.taskLib[id].J = Eigen::MatrixXd::Zero(5, model_nv);
        kin_tasks_walk_.taskLib[id].J(0, 20) = 1;
        kin_tasks_walk_.taskLib[id].J(1, 21) = 1;
        kin_tasks_walk_.taskLib[id].J(2, 22) = 1;
        kin_tasks_walk_.taskLib[id].J(3, 23) = 1;
        kin_tasks_walk_.taskLib[id].J(4, 24) = 1;
        kin_tasks_walk_.taskLib[id].dJ = Eigen::MatrixXd::Zero(5, model_nv);
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("Roll_Pitch_Yaw_Pz");
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(4, 4) * 100;
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(4, 4) * 10;
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("PxPy");
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 1000; // 100
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 50;
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("PosRot");
        if (mode_ == WBCMode::HardContactMode) {
            kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 500;
            kin_tasks_walk_.taskLib[id].kp.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 500;
            kin_tasks_walk_.taskLib[id].kp(0,0) = 800;
            // kin_tasks_walk_.taskLib[id].kp(4,4) = 800;
            // kin_tasks_walk_.taskLib[id].kp(3,3) = 800;
            kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 10;
            kin_tasks_walk_.taskLib[id].kd(4,4) = 10;
            kin_tasks_walk_.taskLib[id].kd(0, 0) = 20;
        } else if (mode_ == WBCMode::SoftContactMode) {
            kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 25.0;
            kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 15.0;
        } 
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("SwingLeg");
        if (mode_ == WBCMode::HardContactMode) {
            kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 500;
            kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 20;
        } else if (mode_ == WBCMode::SoftContactMode) {
            kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 100;
            kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 10;
        } 
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("HandTrackJoints");
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(14);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(14);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(14);
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(14, 14) * 200; // 100
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(14, 14) * 10;
        kin_tasks_walk_.taskLib[id].J = Eigen::MatrixXd::Zero(14, model_nv);
        kin_tasks_walk_.taskLib[id].J.block(0, 6, 14, 14) = Eigen::MatrixXd::Identity(14, 14);
        kin_tasks_walk_.taskLib[id].dJ = Eigen::MatrixXd::Zero(14, model_nv);
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

    }

    // --------------stand------------------
    {
        int id = -1;
        if (mode_ == WBCMode::HardContactMode) {
            id = kin_tasks_stand_.getId("static_Contact");
            kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(12);
            kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(12);
            kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(12);
            kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(12);
            kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(12, 12) * 0;
            kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(12, 12) * 0;
            kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(12, model_nv);
            kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
        }

        id = kin_tasks_stand_.getId("HipRPY");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 1000;
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 50;
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(3, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_stand_.getId("Pz");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(1);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(1);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(1);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(1, 1) * 2000; // 100
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(1, 1) * 10;
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);


        id = kin_tasks_stand_.getId("CoMTrack");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 2000; // 100
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 100;
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(2, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_stand_.getId("CoMXY_HipRPY");
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(5);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(5);
        kin_tasks_stand_.taskLib[id].kp = 
            Eigen::MatrixXd::Identity(5, 5) * 250;
		kin_tasks_stand_.taskLib[id].kp.bottomRightCorner(3, 3).diagonal() *= 4.0;
		kin_tasks_stand_.taskLib[id].kd = 
            Eigen::MatrixXd::Identity(5, 5) * 10.0;
        kin_tasks_stand_.taskLib[id].kd.bottomRightCorner(3, 3).diagonal() *= 10.0;
        if (mode_ == WBCMode::SoftContactMode) {
            kin_tasks_stand_.taskLib[id].kp = 
                Eigen::MatrixXd::Identity(5, 5) * 20.0;
		    kin_tasks_stand_.taskLib[id].kd = 
                Eigen::MatrixXd::Identity(5, 5) * 10.0;
            kin_tasks_stand_.taskLib[id].kd.bottomRightCorner(3, 3).diagonal() *= 10.0;
        }
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(5, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal()(22) = 200;
        kin_tasks_stand_.taskLib[id].W.diagonal()(23) = 200;

        id = kin_tasks_stand_.getId("HandTrackJoints");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(14);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(14);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(14);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(14, 14) * 2000; // 100
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(14, 14) * 100;
        kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(14, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(0, 6, 14, 14) = Eigen::MatrixXd::Identity(14, 14);
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(14, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);


        id = kin_tasks_stand_.getId("HeadRP");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(2);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 100; // 100
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 10;
        kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(2, model_nv);
        kin_tasks_stand_.taskLib[id].J(0, 20) = 1;
        kin_tasks_stand_.taskLib[id].J(1, 21) = 1;
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(2, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_stand_.getId("Roll_Pitch_Yaw");
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 2000;
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 100;
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);


        id = kin_tasks_stand_.getId("fixedWaist");
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 200;
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 20;
        kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(3, model_nv);
        kin_tasks_stand_.taskLib[id].J(0, 22) = 1;
        kin_tasks_stand_.taskLib[id].J(1, 23) = 1;
        kin_tasks_stand_.taskLib[id].J(2, 24) = 1;
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(3, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

    }
}

void WBC_priority::copy_Eigen_to_real_t(
        qpOASES::real_t *target, const Eigen::MatrixXd& source) {
    const int n_rows = source.rows();
    const int n_cols = source.cols();
    int count = 0;
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            target[count] = source(i, j);
            count++;
        }
    }
}

}