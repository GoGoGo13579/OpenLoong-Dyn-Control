#include<mpc_v2.h>
#include<Eigen/Dense>
#include<useful_math.h>

namespace CostomAlgorithm {

using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::NoChange;

constexpr double MaxDouble = 1e8;
constexpr double MinDouble = -1e8;

// !!!!!!!若无特殊声明,所有量均为世界坐标系下
MPC::MPC(const double dt_input) {
    // 机器人相关
    offset_left_foot_to_com_w_.setZero();
    offset_right_foot_to_com_w_.setZero();
    Ic_b_.setZero();
    Rz_f2w_.setIdentity();
    // 状态空间方程常数部分初始化
    dt_ = dt_input;
    Ad_.setIdentity();
    Ad_.block<3, 3>(3, 9) = Matrix<double, 3, 3>::Identity() * dt_;
    Bd_.setZero();
    Bd_.block<3, 3>(9, 0) = Matrix<double, 3, 3>::Identity() * dt_ / m_;
    Bd_.block<3, 3>(9, 9) = Matrix<double, 3, 3>::Identity() * dt_ / m_;
    Gd_.setZero();
    Gd_(state_dim_ - 1, 0) = -1.0 * dt_;
    // 二次规划相关
    Q_bar_.resize(state_dim_ * prediction_horizon_, 
                state_dim_ * prediction_horizon_);
    R_bar_.resize(control_dim_ * control_horizon_, 
                  control_dim_ * control_horizon_);
    X_ref_.resize(state_dim_ * prediction_horizon_);
    U_ref_.resize(control_dim_ * control_horizon_);
    Q_bar_.setIdentity();
    R_bar_.setIdentity();
    Fr_ff_.setZero();
    // 力与力矩的上下界

    control_ub_single_leg_(0) = 1000.0;
    control_ub_single_leg_(1) = 1000.0;
    control_ub_single_leg_(2) = 3.0 * m_ * g_;
    control_ub_single_leg_(3) = 20.0;
    control_ub_single_leg_(4) = 80.0;
    control_ub_single_leg_(5) = 100.0;

    control_lb_single_leg_(0) = -1000.0;
    control_lb_single_leg_(1) = -1000.0;
    control_lb_single_leg_(2) = 0.0;
    control_lb_single_leg_(3) = -20.0;
    control_lb_single_leg_(4) = -80.0;
    control_lb_single_leg_(5) = -100.0;

}

void MPC::dataBusRead(const DataBus &robot_state){
    // 初值状态
    // x = [ψ, p, ω, p_dot]
    x_cur_.block<3, 1>(0, 0) = robot_state.base_rpy;
    x_cur_.block<3, 1>(3, 0) = robot_state.q.block<3, 1>(0, 0);
    x_cur_.block<3, 1>(6, 0) = robot_state.dq.block<3, 1>(3, 0);
    x_cur_.block<3, 1>(9, 0) = robot_state.dq.block<3, 1>(0, 0);
    // 目标状态设置
    if (enable_) {
        // 将Xd_所有时间步前移一位
        for (int i = 0; i < prediction_horizon_ - 1; ++i) {
            X_ref_.block<state_dim_, 1>(i + state_dim_, 0) = 
            X_ref_.block<state_dim_, 1>((i + 1) * state_dim_, 0);
        }
        // 将Xd_最后一个时间步赋值为joystick期望值
        for (int i = 0; i < 3; ++i) {
            X_ref_((prediction_horizon_ - 1) * state_dim_ + i) = 
                robot_state.js_eul_des(i);
            X_ref_((prediction_horizon_ - 1) * state_dim_ + i + 3) = 
                robot_state.js_pos_des(i);
            X_ref_((prediction_horizon_ - 1) * state_dim_ + i + 6) = 
                robot_state.js_omega_des(i);
            X_ref_((prediction_horizon_ - 1) * state_dim_ + i + 9) = 
                robot_state.js_vel_des(i);
        }

    } else {
        for (int i = 0; i < prediction_horizon_; ++i) {
            X_ref_.block<state_dim_, 1>(i * state_dim_, 0) = x_cur_;
        }
    }
    // 足端到质心的位矢，世界坐标系
    offset_left_foot_to_com_w_ = 
        robot_state.fe_l_pos_W - x_cur_.block<3, 1>(3, 0);
    offset_right_foot_to_com_w_ = 
        robot_state.fe_r_pos_W - x_cur_.block<3, 1>(3, 0);
    // 物体系下绕质心的转动惯量
    Ic_b_ = robot_state.inertia;
    // 腿部状态
    DataBus::LegState leg_state_cur = robot_state.legState;
    DataBus::LegState leg_state_next = robot_state.legStateNext;
    for (int i = 0; i < prediction_horizon_; ++i) {
        // 根据相位phi来判断预测时域内腿部的状态
        double delta_phi = i * dt_ / 0.4;
        double phi = robot_state.phi + delta_phi;
        if (phi <= 1.0) {
            leg_state_array_[i] = leg_state_cur;
        } else {
            leg_state_array_[i] = leg_state_next;
        }
    }
    // 旋转矩阵数组.w2f：世界->身体,f2w：身体->世界, Rz_f2w绕z轴的旋转矩阵
    Eigen::Matrix<double, 3, 3> R_slop = 
        eul2Rot(robot_state.slop(0), robot_state.slop(1), robot_state.slop(2));
    for (int i = 0; i < prediction_horizon_; ++i) {
        if (leg_state_array_[i] == DataBus::LSt) {
            R_b2w_array_[i] = robot_state.fe_l_rot_W;
        } else if (leg_state_array_[i] == DataBus::RSt) {
            R_b2w_array_[i] = robot_state.fe_r_rot_W;
        } else {
            R_b2w_array_[i] = R_slop;
        }
        R_w2b_array_[i] = (R_b2w_array_[i]).transpose();
    }

    Rz_f2w_ = Rz3(x_cur_(2));
    // 输入的期望值
    for (int i = 0; i < control_horizon_; ++i) {
        if (leg_state_array_[i] == DataBus::LSt) {
            U_ref_[i * control_dim_ + 2] = m_ * g_;
        } else if (leg_state_array_[i] == DataBus::RSt) {
            U_ref_[i * control_dim_ + 6] = m_ * g_;
        } else {
            U_ref_[i * control_dim_ + 2] = m_ * g_ / 2.0;
            U_ref_[i * control_dim_ + 6] = m_ * g_ / 2.0;
        }
    }
}

void MPC::set_weight(
        const double control_weight_input, 
        const MatrixXd& Q_diag,
        const MatrixXd& R_diag) {
    // X = Aqp*x0 + Bqp*U + Gqp
    // control_lb_single_leg_ X'*Q_bar*X + control_weight*U'*R_bar*U
    // 状态和输入都应该在物体系下惩罚,状态中的前三个分量为欧拉角,不用做坐标变化
    control_weight_ = control_weight_input;
    for (int i = 0; i < prediction_horizon_; ++i) {
        Matrix<double, control_dim_, control_dim_> R_w2b_extended =
            Matrix<double, control_dim_, control_dim_>::Identity();
        R_w2b_extended.block<3, 3>(3, 3) = R_w2b_array_[i];
        R_w2b_extended.block<3, 3>(6, 6) = R_w2b_array_[i];
        R_w2b_extended.block<3, 3>(9, 9) = R_w2b_array_[i];
        for (int j = 0; j < state_dim_; ++j) {
            Q_bar_(i * state_dim_ + j, i * state_dim_ + j) = Q_diag(j);
        }
        Q_bar_.block<state_dim_, state_dim_>(i * state_dim_, i * state_dim_) =
            R_w2b_extended.transpose() *
            Q_bar_.block<state_dim_, state_dim_>(i * state_dim_, i * state_dim_) *
            R_w2b_extended;
    }
    for (int i = 0; i < control_horizon_; ++i) {
        Matrix<double, control_dim_, control_dim_> R_w2b_extended =
            Matrix<double, control_dim_, control_dim_>::Zero();
        R_w2b_extended.block<3, 3>(0, 0) = R_w2b_array_[i];
        R_w2b_extended.block<3, 3>(3, 3) = R_w2b_array_[i];
        R_w2b_extended.block<3, 3>(6, 6) = R_w2b_array_[i];
        R_w2b_extended.block<3, 3>(9, 9) = R_w2b_array_[i];
        for (int j = 0; j < control_dim_; ++j) {
            R_bar_(i * control_dim_ + j, i * control_dim_ + j) = R_diag(j);
        }
        R_bar_.block<control_dim_, control_dim_>(i * control_dim_, i * control_dim_) =
            R_w2b_extended.transpose() * 
            R_bar_.block<control_dim_, control_dim_>(i * control_dim_, i * control_dim_) *
            R_w2b_extended;
    }
    
}

void MPC::cal() {
    if(!enable_) {
        return;
    }
    // 更新状态矩阵与输入矩阵中可变部分
    Ad_.block<3, 3>(0, 6) = Rz_f2w_ * dt_;
    // 全局坐标系下绕质心的转动惯量
    Matrix<double, 3, 3> Ic_w_inv = (Rz_f2w_ * Ic_b_ * Rz_f2w_.transpose()).inverse();
    Bd_.block<3, 3>(6, 0) = 
        Ic_w_inv * CrossProduct_A(offset_left_foot_to_com_w_) * dt_;
    Bd_.block<3, 3>(6, 3) = Ic_w_inv;
    Bd_.block<3, 3>(6, 6) = 
        Ic_w_inv * CrossProduct_A(offset_right_foot_to_com_w_) * dt_;

    // X = Aqp*x0 + Bqp*U + Gqp
    // Aqp. 
    std::array<Matrix<double, state_dim_, state_dim_>, 
               prediction_horizon_> Ad_power_array;
    Ad_power_array[0] = Ad_;
    for (int i = 1; i < prediction_horizon_; ++i) {
        Ad_power_array[i] = Ad_ * Ad_power_array[i - 1];
    }

    Matrix<double, Dynamic, state_dim_> Aqp;
    Aqp.resize(state_dim_ * prediction_horizon_, NoChange);
    for (int i = 0; i < prediction_horizon_; ++i) {
        Aqp.block<state_dim_, state_dim_>(i * state_dim_, i * state_dim_)  = 
            Ad_power_array[i];
    }

    // Gqp
    Matrix<double, Dynamic, 1> Gqp;
    Gqp.resize(state_dim_ * prediction_horizon_, NoChange);
    Gqp.block<state_dim_, state_dim_>(0, 0) = 
        Matrix<double, state_dim_, state_dim_>::Identity();
    for (int i = 1; i < prediction_horizon_; ++i) {
        Gqp.block<state_dim_, state_dim_>(i * state_dim_, i * state_dim_ ) = 
            Gqp.block<state_dim_, state_dim_>((i - 1) * state_dim_, (i - 1) * state_dim_) +
            Ad_power_array[i - 1];
    }

    // Bqp 由于采用控制时域与预测时域独立,Bqp相较于两种时域全是N的情况有所变化
    // 原理可参考我的笔记或者https://aleksandarhaber.com/中关于MPC的教程
    std::array<Matrix<double, state_dim_, control_dim_>, 
               prediction_horizon_> Ad_power_Bd_array;
    Ad_power_Bd_array[0] = Bd_;
    for (int i = 1; i < prediction_horizon_; ++i) {
        Ad_power_Bd_array[i] = Ad_ * Ad_power_Bd_array[i - 1];
    }
    Matrix<double, Dynamic, Dynamic> Bqp;
    Bqp.resize(state_dim_ * prediction_horizon_, control_dim_ * control_horizon_);
    Bqp.setZero();
    for (int i = 0; i < prediction_horizon_; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            const int power = i - j;
            if (j < control_horizon_ - 1) {
                Bqp.block<state_dim_, control_dim_>(i * state_dim_, j * control_dim_) = 
                Ad_power_Bd_array[power];
            } else if (j == control_horizon_ - 1) {
                for (int k = 0; k <= power; ++k) {
                    Bqp.block<state_dim_, control_dim_>
                        (i * state_dim_, j * control_dim_) += 
                    Ad_power_Bd_array[k];
                }
            } else if (j > control_horizon_ - 1) {
                break;
            }
        }
    }

    // 求解二次规划
    // 目标函数
    MatrixXd H;
    H = 2 * (Bqp.transpose() * Q_bar_ * Bqp + R_bar_);
    VectorXd g;
    g = 2 * (Bqp.transpose() * Q_bar_ * (Aqp * x_cur_ + Gqp - X_ref_)) -
        2 * R_bar_ * U_ref_;

    // 约束条件
    // 全部约束统一为 Ax <= 0 形式
    // 单条腿的约束
    constexpr int nc_single_leg = 10;
    constexpr int control_dim_single_leg =  control_dim_ / 2;
    Matrix<double, nc_single_leg, control_dim_single_leg> A_single_leg_block;
    // 摩擦锥约束,fz大于放在lb和ub里
    A_single_leg_block.block<4, 3>(0, 0) <<  
            -1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
            1.0, 0.0, -1.0 / sqrt(2.0) * miu_,
            0.0, -1.0, -1.0 / sqrt(2.0) * miu_,
            0.0, 1.0, -1.0 / sqrt(2.0) * miu_;
    // 物体系下z方向力矩约束 |mz| <= factor * μ * fz
    // 旋转矩阵最后统一处理
    constexpr int factor = 0.6; // 这个系数的计算可以参考笔记
    A_single_leg_block.block<1, 6>(4, 0) <<
        0.0, 0.0, -factor * miu_, 0.0, 0.0, 1.0;
    A_single_leg_block.block<1, 6>(5, 0) <<
        0.0, 0.0, -factor * miu_, 0.0, 0.0, -1.0;
    // 物体系下x, y方向力矩约束
    // 物理意义：由mx与my造成的zmp偏移不要太大
    constexpr double feet_half_x_length = 0.1225;
    constexpr double feet_half_y_length = 0.04;
    constexpr double factor2 = 0.8;
    A_single_leg_block.block<4, 6>(6, 0) <<
        0.0, 0.0, -factor2 * feet_half_x_length, 0.0, 1.0, 0.0,
        0.0, 0.0, -factor2 * feet_half_x_length, 0.0, -1.0, 0.0,
        0.0, 0.0, -factor2 * feet_half_y_length, 0.0, 1, 0.0,
        0.0, 0.0, -factor2 * feet_half_x_length, 0.0, -1, 0.0;

    // 汇总
    // -∞ <= Ax <= 0
    MatrixXd A;
    A.resize(nc_single_leg * 2 * control_horizon_, 
             control_dim_ * control_horizon_);
    A.setZero();
    VectorXd lbA, ubA;
    lbA.resize(nc_single_leg * 2 * control_horizon_);
    ubA.resize(nc_single_leg * 2 * control_horizon_);
    lbA.setConstant(MinDouble);
    ubA.setConstant(MaxDouble);
    for (int i = 0; i < control_horizon_; ++i) {
        // 控制时域内每一个时间步的左右脚
        A.block<nc_single_leg, control_dim_single_leg>
            (2*i * nc_single_leg, i * control_dim_) = A_single_leg_block;
        A.block<nc_single_leg, control_dim_single_leg>
            ((2*i + 1) * nc_single_leg, i * control_dim_) = A_single_leg_block;
        // 关于力矩的约束要在物体系下表达，因此部分约束要做坐标变化
        Matrix<double, 6, 6> R_extended_w2b = Matrix<double, 6, 6>::Zero();
        R_extended_w2b.block<3, 3>(0, 0) = R_w2b_array_[i];
        R_extended_w2b.block<3, 3>(3, 3) = R_w2b_array_[i];
        A.block<6, control_dim_single_leg>(2*i * nc_single_leg + 4, i * control_dim_) *= 
            R_extended_w2b;
        A.block<6, control_dim_single_leg>((2*i + 1) * nc_single_leg + 4, i * control_dim_) *=
            R_extended_w2b;  

        // 根据legstate设置上界
        if (leg_state_array_[i] == DataBus::LSt) {
            ubA.block<nc_single_leg, 1>(2*i * control_dim_single_leg, 0).setZero();
        } else if (leg_state_array_[i] == DataBus::RSt) {
            ubA.block<nc_single_leg, 1>((2 * i + 1) * control_dim_single_leg, 0).setZero();
        } else {
            ubA.block<2 * nc_single_leg, 1>(i * control_dim_, 0).setZero();
        }
    }

    // 优化变量上下界
    VectorXd lb, ub;
    lb.resize(control_dim_ * control_horizon_);
    ub.resize(control_dim_ * control_horizon_);
    for (int i = 0; i < control_horizon_; ++i) {
        lb.block<control_dim_, 1>(i * control_dim_, 0) << 
            control_ub_single_leg_, control_lb_single_leg_;
        ub.block<control_dim_, 1>(i * control_dim_, 0) <<
            control_ub_single_leg_, control_ub_single_leg_;
    }

    // 猜测初始解
    VectorXd U_opt_ini_guess;
    U_opt_ini_guess.resize(control_dim_ * control_horizon_);
    for (int i = 0; i < control_horizon_; ++i) {
        if (leg_state_array_[i] == DataBus::LSt) {
            U_opt_ini_guess(i * control_dim_ + 2) = m_ * g_;
        } else if (leg_state_array_[i] == DataBus::RSt) {
            U_opt_ini_guess(i * control_dim_ + 8) = m_ * g_;
        } else {
            U_opt_ini_guess(i * control_dim_ + 2) = m_ * g_ / 2.0;
            U_opt_ini_guess(i * control_dim_ + 8) = m_ * g_ / 2.0;
        }
    }

    // 求解QP
    // 形式转化
    qpOASES::real_t *H_qpoases = new qpOASES::real_t[H.size()];  
    qpOASES::real_t *g_qpoases = new qpOASES::real_t[g.size()];  
    qpOASES::real_t *A_qpoases = new qpOASES::real_t[A.size()];  
    qpOASES::real_t *ubA_qpoases = new qpOASES::real_t[ubA.size()]; 
    qpOASES::real_t *lbA_qpoases = new qpOASES::real_t[lbA.size()]; 
    qpOASES::real_t *ub_qpoases = new qpOASES::real_t[ub.size()];  
    qpOASES::real_t *lb_qpoases = new qpOASES::real_t[lb.size()]; 
    qpOASES::real_t *U_opt_ini_guess_qpoases = 
        new qpOASES::real_t[U_opt_ini_guess.size()]; 
    copy_Eigen_to_real_t(H_qpoases, H);
    copy_Eigen_to_real_t(g_qpoases, g);
    copy_Eigen_to_real_t(A_qpoases, A);
    copy_Eigen_to_real_t(ubA_qpoases, ubA);
    copy_Eigen_to_real_t(lbA_qpoases, lbA);
    copy_Eigen_to_real_t(ub_qpoases, ub);
    copy_Eigen_to_real_t(lb_qpoases, lb);
    copy_Eigen_to_real_t(U_opt_ini_guess_qpoases, U_opt_ini_guess);
    qp_nWSR_ = 1000;

    qpOASES::Options option;
    option.setToMPC();
    option.printLevel = qpOASES::PL_LOW;
    qpOASES::QProblem qp_solver(H.rows(), A.rows());
    qp_solver.setOptions(option);
    auto qp_status_ = qp_solver.init(
        H_qpoases, g_qpoases, A_qpoases, 
        lb_qpoases, ub_qpoases, lbA_qpoases, 
        ubA_qpoases, qp_nWSR_, &qp_cpu_time_, U_opt_ini_guess_qpoases);

    if (qp_status_ = qpOASES::returnValue::SUCCESSFUL_RETURN) {
        qpOASES::real_t* U_opt;
        qp_solver.getPrimalSolution(U_opt);
        for (int i = 0; i < control_dim_; ++i) {
            Fr_ff_(i) = U_opt[i];
        }
        Fr_ff_pre_ = Fr_ff_;
    } else {
        Fr_ff_ = Fr_ff_pre_;
    }

    delete[] H_qpoases;
    delete[] g_qpoases;
    delete[] A_qpoases;
    delete[] ubA_qpoases;
    delete[] lbA_qpoases;
    delete[] ub_qpoases; 
    delete[] lb_qpoases;

}

void MPC::enable () {
    enable_ = true;
}

void MPC::disable () {
    enable_ = false;
}

bool MPC::get_ENA () const{
    return enable_;
}

void MPC::dataBusWrite(DataBus &robot_state) const {
    robot_state.Fr_ff = Fr_ff_;
    robot_state.qpStatus_MPC = qp_status_;
    robot_state.qp_cpuTime_MPC = qp_cpu_time_;
    robot_state.qp_nWSR_MPC = qp_nWSR_;
}

void MPC::copy_Eigen_to_real_t(
        qpOASES::real_t *target, const MatrixXd& source) {
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

} // end namespace CostomAlgorithm


    