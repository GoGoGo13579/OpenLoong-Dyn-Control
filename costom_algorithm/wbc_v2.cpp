#include"wbc_v2.h"
#include<qpOASES.hpp>

namespace CustomAlgorithm {

constexpr double BigDouble = 1e10;

WBC_priority::WBC_priority(
        const double timestep_input, 
        const int model_nv) {
    timestep_ = timestep_input;
    initTasks(model_nv);
    // 足端力矩上下界,物体系下
    tau_upp_stand_L_ << 15, 30, 40;
    tau_low_stand_L_ << -15, -30, -40;
    tau_upp_walk_L_ << 15, 40, 40; 
    tau_low_walk_L_ << -15, -40, -40;

    delta_q_final_kin_ = Eigen::VectorXd::Zero(model_nv);
    dq_final_kin_ = Eigen::VectorXd::Zero(model_nv);
    ddq_final_kin_ = Eigen::VectorXd::Zero(model_nv);
    ddp_final_dyn_ = Eigen::VectorXd::Zero(model_nv);
    tau_final_dyn_ = Eigen::VectorXd::Zero(model_nv - 6);
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
        int id = kin_tasks_walk_.getId("static_Contact");
        kin_tasks_walk_.taskLib[id].J = J_contact;
        kin_tasks_walk_.taskLib[id].dJ = dJ_contact;

        // 任务：冗余关节尽量不动，冗余关节指的是头部（2个），腰（3个）
        id = kin_tasks_walk_.getId("RedundantJoints");
        kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(5);
        kin_tasks_walk_.taskLib[id].errX(0) = 0 - robot_state_ptr_->q(21);
        kin_tasks_walk_.taskLib[id].errX(1) = 0 - robot_state_ptr_->q(22);
        kin_tasks_walk_.taskLib[id].errX(2) = 0 - robot_state_ptr_->q(23);
        kin_tasks_walk_.taskLib[id].errX(3) = 0 - robot_state_ptr_->q(24);
        kin_tasks_walk_.taskLib[id].errX(4) = 0 - robot_state_ptr_->q(25);
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(5);

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
        if (fabs(kin_tasks_walk_.taskLib[id].errX(0)) >= 0.02)
            kin_tasks_walk_.taskLib[id].errX(0) = 0.02 * sign(kin_tasks_walk_.taskLib[id].errX(0));
        if (fabs(kin_tasks_walk_.taskLib[id].errX(1)) >= 0.02)
            kin_tasks_walk_.taskLib[id].errX(1) = 0.02 * sign(kin_tasks_walk_.taskLib[id].errX(1));
        if (kin_tasks_walk_.taskLib[id].errX(2)>0.005){
            kin_tasks_walk_.taskLib[id].errX(2) = 0.005;
        }
        //  求出来是一个so3
        desRot = eul2Rot(robot_state_ptr_->base_rpy_des(0), robot_state_ptr_->base_rpy_des(1), robot_state_ptr_->base_rpy_des(2));
        kin_tasks_walk_.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(robot_state_ptr_->base_rot, desRot);
        kin_tasks_walk_.taskLib[id].errX(4) -= 0.05 * robot_state_ptr_->dq(4);
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
        // kin_tasks_walk_.taskLib[id].derrX = des_dq.block(0, 0, 6, 1) - dq.block(0, 0, 6, 1);
        // kin_tasks_walk_.taskLib[id].derrX << (robot_state_ptr_->base_vel_des - base_vel_cur),
        //                                     (base_omega_des - base_omega_cur);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
        // kin_tasks_walk_.taskLib[id].dxDes << robot_state_ptr_->base_vel_des, base_omega_des;
        kin_tasks_walk_.taskLib[id].dxDes.block(0, 0, 3, 1) = robot_state_ptr_->base_vel_des;
        kin_tasks_walk_.taskLib[id].J = robot_state_ptr_->J_base;
        kin_tasks_walk_.taskLib[id].dJ = robot_state_ptr_->dJ_base;

        id = kin_tasks_walk_.getId("SwingLeg");
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
        //        kin_tasks_walk_.taskLib[id].derrX=-J_swing*dq;
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
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
            robot_state_ptr_->q.block(0, 0, 3, 1).block(0, 0, 2, 1) - 
            robot_state_ptr_->pCoM_W.block(0, 0, 2, 1);
		desRot = eul2Rot(robot_state_ptr_->base_rpy_des(0),
                        robot_state_ptr_->base_rpy_des(1), 
                        robot_state_ptr_->base_rpy_des(2));
        kin_tasks_stand_.taskLib[id].errX.block<3, 1>(2, 0) = 
            diffRot(robot_state_ptr_->hip_link_rot, desRot);
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(5);
//                    kin_tasks_stand_.taskLib[id].derrX.block(0,0,2,1)=-(robot_state_ptr_->Jcom_W*dq).block(0,0,2,1);
//                    kin_tasks_stand_.taskLib[id].derrX.block(2,0,3,1)=-taskMapRPY*robot_state_ptr_->J_hip_link*dq;
        kin_tasks_stand_.taskLib[id].J = Eigen::MatrixXd::Zero(5, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(0, 0, 2, model_nv) = robot_state_ptr_->Jcom_W.block(0, 0, 2, model_nv);
        kin_tasks_stand_.taskLib[id].J.block(2, 0, 3, model_nv) = taskMapRPY * robot_state_ptr_->J_hip_link;
        kin_tasks_stand_.taskLib[id].J.block(2, 22, 3, 3).setZero(); // exculde waist joints
        kin_tasks_stand_.taskLib[id].J.block(2, 6, 3, 14).setZero(); // exculde arm joints

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
    // 18 means the sum of dims of delta_r and delta_Fr
    const int variable_dim = 18;
    const int constraint_dim = 22;
    // 等式约束 在qpoases中等式约束的表达形式为 b <= A <= b
    // 将浮动基动力学，加速度松弛，反作用力松弛统一表达为一个等式约束
    /// 具体形式可参考笔记
    // 浮动基选择矩阵，浮动基加速度指令选择矩阵
    Eigen::MatrixXd Selector_flaot_Mat, Selector_delatf_Mat, J_foot_end; 
    Selector_flaot_Mat.setZero(6, robot_state_ptr_->model_nv);
    Selector_flaot_Mat.block(0, 0, 6, 6).setIdentity();
    Selector_delatf_Mat.setZero(robot_state_ptr_->model_nv, 6);
    Selector_delatf_Mat.block(0, 0, 6, 6).setIdentity();
    J_foot_end.resize(12, robot_state_ptr_->model_nv);
    J_foot_end.block(0, 0, 6, robot_state_ptr_->model_nv) =
        robot_state_ptr_->J_l;
    J_foot_end.block(6, 0, 6, robot_state_ptr_->model_nv) =
        robot_state_ptr_->J_r;

    Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(6, variable_dim); 
    A1.block<6, 6>(0, 0) = 
        Selector_flaot_Mat * robot_state_ptr_->dyn_M * Selector_delatf_Mat;
    A1.block<6, 12>(0, 6) = -Selector_flaot_Mat * J_foot_end.transpose();

    Eigen::Vector<double, 6> ubA1 = Eigen::VectorXd::Zero(6);
    Eigen::Vector<double, 6> lbA1 = Eigen::VectorXd::Zero(6);
    ubA1 = 
        Selector_flaot_Mat * J_foot_end.transpose() * robot_state_ptr_->Fr_ff -
        Selector_flaot_Mat * robot_state_ptr_->dyn_M * ddq_final_kin_ - Selector_flaot_Mat * robot_state_ptr_->dyn_Non ;
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
            Eigen::Vector4d::Zero(), f_z_upp_, tau_upp_fe;
        f_low.block(start, 0, 8, 1) << 
            Eigen::Vector4d::Constant(-BigDouble), f_z_low_, tau_low_fe;
    }

    Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(16, 18);
    A2.block<16, 12>(0, 6) = A2_block;
    Eigen::VectorXd lbA2 = Eigen::VectorXd::Zero(16);
    Eigen::VectorXd ubA2 = Eigen::VectorXd::Zero(16);

    lbA2 = f_low - A2_block * robot_state_ptr_->Fr_ff;
    ubA2 = f_upp - A2_block * robot_state_ptr_->Fr_ff;

    // 将等式约束与不等式约束拼接成总的约束
    Eigen::MatrixXd A_final = 
        Eigen::MatrixXd::Zero(constraint_dim, variable_dim);
    A_final.block<6, 18>(0, 0) = A1;
    A_final.block<16, 18>(6, 0) = A2;

    Eigen::VectorXd lbA_final = Eigen::VectorXd::Zero(constraint_dim);
    Eigen::VectorXd ubA_final = Eigen::VectorXd::Zero(constraint_dim);

    lbA_final.block<6, 1>(0, 0) = lbA1;
    lbA_final.block<16, 1>(6, 0) = lbA2;
    ubA_final.block<6, 1>(0, 0) = ubA1;
    ubA_final.block<16, 1>(6, 0) = ubA2;

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
    qpOASES::real_t *lbA_qpoases = new qpOASES::real_t[constraint_dim];
    qpOASES::real_t *ubA_qpoases = new qpOASES::real_t[constraint_dim];

    copy_Eigen_to_real_t(H_qpoases, H);
    copy_Eigen_to_real_t(A_qpoases, A_final);
    copy_Eigen_to_real_t(lbA_qpoases, lbA_final);
    copy_Eigen_to_real_t(ubA_qpoases, ubA_final);
    for (int i = 0; i < variable_dim; i++) {
        g_qpoases[i] = 0;
    }
    std::cout << std::endl;
    nWSR_ = 200;
    cpu_time_ = timestep_;

    qpOASES::QProblem QP_prob(H.rows(), A_final.rows());
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
            robot_state_ptr_->dyn_Non - J_foot_end.transpose() * Fr_final_dyn_;

    tau_final_dyn_ = tauRes.block(6, 0, robot_state_ptr_->model_nv - 6, 1);

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

    taskOrder_walk.emplace_back("static_Contact");
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

    taskOrder_stand.emplace_back("static_Contact");
    taskOrder_stand.emplace_back("CoMXY_HipRPY");
    taskOrder_stand.emplace_back("Pz");
    taskOrder_stand.emplace_back("HandTrackJoints");
    taskOrder_stand.emplace_back("HeadRP");

    kin_tasks_stand_.buildPriority(taskOrder_stand);

    // 2.静态部分赋值
    // ------------walk-------------
    {
        int id = kin_tasks_walk_.getId("static_Contact");
        kin_tasks_walk_.taskLib[id].errX = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 0;
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 0;
        kin_tasks_walk_.taskLib[id].W.diagonal() = 
            Eigen::VectorXd::Ones(model_nv);

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
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 500;
        kin_tasks_walk_.taskLib[id].kp.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 500;
        kin_tasks_walk_.taskLib[id].kp(0,0) = 800;
        // kin_tasks_walk_.taskLib[id].kp(4,4) = 800;
        // kin_tasks_walk_.taskLib[id].kp(3,3) = 800;
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 10;
        kin_tasks_walk_.taskLib[id].kd(4,4) = 10;
        kin_tasks_walk_.taskLib[id].kd(0, 0) = 20;
        kin_tasks_walk_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

        id = kin_tasks_walk_.getId("SwingLeg");
        kin_tasks_walk_.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 500;
        kin_tasks_walk_.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 20;
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
        int id = kin_tasks_stand_.getId("static_Contact");
        kin_tasks_stand_.taskLib[id].errX = Eigen::VectorXd::Zero(12);
        kin_tasks_stand_.taskLib[id].derrX = Eigen::VectorXd::Zero(12);
        kin_tasks_stand_.taskLib[id].ddxDes = Eigen::VectorXd::Zero(12);
        kin_tasks_stand_.taskLib[id].dxDes = Eigen::VectorXd::Zero(12);
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(12, 12) * 0;
        kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(12, 12) * 0;
        kin_tasks_stand_.taskLib[id].dJ = Eigen::MatrixXd::Zero(12, model_nv);
        kin_tasks_stand_.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

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
        kin_tasks_stand_.taskLib[id].kp = Eigen::MatrixXd::Identity(5, 5) * 250; // 100
		kin_tasks_stand_.taskLib[id].kp.block(2, 2, 3, 3) = Eigen::MatrixXd::Identity(3,3)*1000;
		kin_tasks_stand_.taskLib[id].kd = Eigen::MatrixXd::Identity(5, 5) * 10;
        	kin_tasks_stand_.taskLib[id].kd.block(2, 2, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 10;   // 100  // for hip rpy
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