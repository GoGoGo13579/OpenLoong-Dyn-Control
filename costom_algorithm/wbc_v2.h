#pragma once

#include <Eigen/Dense>
#include "data_bus.h"
#include "pino_kin_dyn.h"
#include "priority_tasks_v2.h"
#include "qpOASES.hpp"

namespace CostomAlgorithm {

enum class WBCMode {
    HardContactMode = 0, //接触作为最高优先级任务
    SoftContactMode = 1  //接触不作为最高优先级任务，由二次规划保证接触
}; 

class WBC_priority {
 public:
    WBC_priority(const double timestep_input, const int model_nv);
    ~WBC_priority() = default;
    // 读取数据
    void dataBusRead(const DataBus &robotState);
    void dataBusRead(const DataBus *robot_state_input);
    // 输出结果
    void dataBusWrite(DataBus &robotState);
    // 模式设置
    // 模式1：最高优先级任务为接触
    // 模式2：接触任务被置于二次规划中解决
    void setMode(const WBCMode& mode_input);
    WBCMode getMode() const;
    // 运动学WBC计算deltaq，dq，ddq指令
    void computeDdq(Pin_KinDyn &pinKinDynIn);
    // 动力学WBC计算
    void computeTau();
 private:
    int model_nv_ = 0;
    WBCMode mode_ = WBCMode::HardContactMode;
    double timestep_ = 0.001;
    double miu_ = 0.5;
    // 数据结构体指针
    const DataBus *robot_state_ptr_ = nullptr;
    // 任务数组
    PriorityTasks kin_tasks_walk_, kin_tasks_stand_;
    // 运动学WBC计算结果
    Eigen::MatrixXd delta_q_final_kin_, dq_final_kin_, ddq_final_kin_;
    // 反作用力上下界
    double fr_z_min_ = 0.0;
    double fr_z_max_ = 2000.0;
    // 力矩上下界
    Eigen::Vector3d tau_upp_stand_L_, tau_low_stand_L_, 
                    tau_upp_walk_L_, tau_low_walk_L_;
    // 初始化任务队列
    void initTasks(const int model_nv);
    // Eigen格式转化为qpoases格式
    void copy_Eigen_to_real_t(
        qpOASES::real_t *target, const Eigen::MatrixXd& source);
    int nWSR_ = 200;
    qpOASES::real_t cpu_time_ = 0.05;
    qpOASES::returnValue qp_status_;
    // 动力学WBC计算结果
    Eigen::VectorXd optimal_var_, optimal_var_pre_,
                    ddp_final_dyn_, Fr_final_dyn_, 
                    tau_final_dyn_;
    // 按照硬接触来计算tau
    void computeTauHardContact();
    // 按照软接触来计算tau
    void computeTauSoftContact();
    Eigen::MatrixXd J_contact_pre_;
    bool is_first_run_ = true;
};

} // end namespace CostomAlgorithm
