#pragma once

#include<Eigen/Dense>
#include<data_bus.h>
#include<qpOASES.hpp>
#include<array>

namespace CustomAlgorithm {

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class MPC {
 public:
    MPC() = default;
    MPC(const double dt_input);
    ~MPC() = default;
    // 读取机器人状态
    void dataBusRead(const DataBus &robot_state);
    // 将运算结果写入机器人状态
    void dataBusWrite(DataBus &robot_state) const;
    // 设置权重
    void set_weight(const double control_weight_input,
                    const MatrixXd& Q_diag,
                    const MatrixXd& R_diag);
    // 允许使用MPC控制器
    void enable();
    // 不允许使用MPC控制器
    void disable();
    // 获取是否使用MPC控制器的标志位
    bool get_ENA() const;
    // 计算MPC控制率
    void cal();
 private:
    // !!!!!!!若无特殊声明,均为世界坐标系下变量
    // ------MPC相关
    // 离散时间步长
    double dt_;
    // 预测时域长度。在设置矩阵模板参数时需要使用，因此在编译期就要确定其值
    // 所以声明为static，其余声明为static的成员变量同理。
    static constexpr int prediction_horizon_ = 2;
    // 控制时域长度
    static constexpr int control_horizon_ = 2;
    // 状态量维度
    static constexpr int state_dim_ = 12;
    // 控制量维度
    static constexpr int control_dim_ = 12;
    // 离散状态空间方程 x(k+1) = Ad*x(k) + Bd*u(k) + Gd
    // x = [ψ, p, ω, p_dot], u = [fl, ml, fr, mr]
    Matrix<double, state_dim_, state_dim_> Ad_;
    Matrix<double, state_dim_, control_dim_> Bd_;
    Matrix<double, state_dim_, 1> Gd_; // 重力项
    // 初始状态
    Vector<double, state_dim_> x_cur_;
    // 期望状态X,期望输入U
    VectorXd X_ref_;
    VectorXd U_ref_;
    // 是否启用MPC的标志位
    bool enable_ = false;
    // 控制输入权重
    double control_weight_ = 1;
    Vector<double, control_dim_ / 2> control_ub_single_leg_, 
                                     control_lb_single_leg_;


    // -------二次规划相关
    // X = Aqp*x0 + Bqp*U + Gqp
    // min X'*Q_bar*X + control_weight*U'*R_bar*U
    MatrixXd Q_bar_;
    MatrixXd R_bar_;
    // 二次规划求解的返回值
    qpOASES::returnValue qp_status_;
    int qp_nWSR_ = 1000;
    qpOASES::real_t qp_cpu_time_;
    // 最优地面反作用力
    Vector<double, control_dim_> Fr_ff_, Fr_ff_pre_;


    // ------机器人相关
    // 质量，重力加速度，摩擦系数
    const double m_ = 77.35;
    const double g_ = 9.8;
    const double miu_ = 0.5;
    // 足端到质心的位矢，世界坐标系
    Vector<double, 3> offset_left_foot_to_com_w_; // 左脚
    Vector<double, 3> offset_right_foot_to_com_w_; // 右脚 
    // 转动惯量,物体系下
    Matrix<double, 3, 3> Ic_b_;
    // 腿部状态,其中leg_states
    std::array<DataBus::LegState, prediction_horizon_> leg_state_array_;
    // 旋转矩阵.w2f：世界->身体,f2w：身体->世界
    std::array<Matrix<double, 3, 3>, prediction_horizon_> R_w2b_array_, R_b2w_array_;
    Matrix<double, 3, 3> Rz_f2w_;

    void copy_Eigen_to_real_t(
        qpOASES::real_t *target, const MatrixXd& source);
};

};

