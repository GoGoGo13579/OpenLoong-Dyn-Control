#pragma once
#include <Eigen/Dense>
#include "useful_math.h"
#include <utility>
#include <vector>
#include <string>
#include <iostream>


namespace CustomAlgorithm {
    
struct Task{
    std::string taskName;
    int id;
    int parentId, childId;
    Eigen::VectorXd dxDes,ddxDes;
    Eigen::VectorXd delta_q, dq, ddq;
    Eigen::MatrixXd J, dJ, Jpre;
    Eigen::MatrixXd N;
    Eigen::MatrixXd kp, kd;
    Eigen::DiagonalMatrix<double,-1> W; // weighted matrix for pseudo inverse
    Eigen::VectorXd errX, derrX;
    Task(std::string name){taskName=name;};
};
class PriorityTasks {
public:
    std::vector<Task> taskLib;
    std::vector<std::string> nameList;
    std::vector<int> idList, parentIdList, childIdList;
    Eigen::VectorXd out_delta_q, out_dq, out_ddq;
    int startId;
    void addTask(const char* name);
    int getId(const std::string& name);
    int getId(const char* name);
    void buildPriority(const std::vector<std::string> &taskOrder);
    void computeAll(
        const Eigen::MatrixXd &dyn_M_inv,
        const Eigen::VectorXd &dq);
    void printTaskInfo();
};

} // end namespace CustomAlgorithm