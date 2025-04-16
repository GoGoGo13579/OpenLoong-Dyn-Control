#include"priority_tasks_v2.h"
#include"costom_utils.hpp"

namespace CostomAlgorithm {

void PriorityTasks::addTask(const char* name) {
    taskLib.emplace_back(name);
    taskLib.back().id=(int)(taskLib.size())-1;
    nameList.emplace_back(name);
}

int PriorityTasks::getId(const std::string &name) {
    for (size_t i = 0; i < nameList.size(); ++i) {
        if (nameList[i]==name) {
            return i;
        }
    }
    return -1;
}

int PriorityTasks::getId(const char* name){
    for (size_t i = 0; i < nameList.size(); ++i) {
        if (nameList[i]==name) {
            return i;
        }
    }
    printf("Cannot find wbc task: %s !\n", name);
    return -1;
}

void PriorityTasks::buildPriority(const std::vector<std::string> &taskOrder) {
    startId= getId(taskOrder[0]);
    for (int i=0;i<taskOrder.size();i++)
    {
        int idCur= getId(taskOrder[i]);
        if (i==0)
            taskLib[idCur].parentId=-1;
        else
        {
            int idBef= getId(taskOrder[i-1]);
            taskLib[idCur].parentId=idBef;
        }
        if (i==taskOrder.size()-1)
            taskLib[idCur].childId=-1;
        else
        {
            int idNxt= getId(taskOrder[i+1]);
            taskLib[idCur].childId=idNxt;
        }
    }
}

void PriorityTasks::printTaskInfo() {
    for (int i=0; i<taskLib.size(); i++)
    {
        printf("-------------\n");
        printf("taskName=%s\n", taskLib[i].taskName.c_str());
        printf("parentId=%d, childId=%d\n", taskLib[i].parentId, taskLib[i].childId);
    }
}

void PriorityTasks::computeAll(
        const Eigen::MatrixXd &dyn_M_inv,
        const Eigen::VectorXd &dq) {
    int curId = startId;
    int parentId = taskLib[curId].parentId;
    int childId = taskLib[curId].childId;
    for (int i=0; i<taskLib.size(); i++)
    {
        if (parentId==-1) {
            taskLib[curId].N =
                Eigen::MatrixXd::Identity(taskLib[curId].J.cols(),
                                            taskLib[curId].J.cols());
            taskLib[curId].Jpre = taskLib[curId].J * taskLib[curId].N;
            taskLib[curId].delta_q =
                pseudoInv_right_weighted(taskLib[curId].Jpre,taskLib[curId].W) *
                taskLib[curId].errX;
            taskLib[curId].dq =
                pseudoInv_right_weighted(taskLib[curId].Jpre,taskLib[curId].W) *
                taskLib[curId].dxDes;
            Eigen::VectorXd ddxcmd =
                taskLib[curId].ddxDes +
                taskLib[curId].kp * taskLib[curId].errX +
                taskLib[curId].kd*taskLib[curId].derrX;
            taskLib[curId].ddq =
                dyn_pseudoInv(taskLib[curId].Jpre,dyn_M_inv,true) *
                (ddxcmd - taskLib[curId].dJ * taskLib[curId].dq);
            // std::cout << std::endl << "=============PriorityTask===========" <<
            //     std::endl;
            // CostomUtils::print_vector(taskLib[curId].delta_q, 
            //     taskLib[curId].taskName + ": delta_q");
            // CostomUtils::print_vector(taskLib[curId].dq,
            //     taskLib[curId].taskName + ": dq");
            // CostomUtils::print_vector(taskLib[curId].ddq,
            //     taskLib[curId].taskName + ": ddq");
            // CostomUtils::print_vector(ddxcmd,
            //     taskLib[curId].taskName + ": ddxcmd");
            // CostomUtils::print_vector(dq,
            //     taskLib[curId].taskName + ": 当前机器人实际dq");
            // CostomUtils::print_matrix(taskLib[curId].kp, 
            //     taskLib[curId].taskName + ": kd");
            // CostomUtils::print_vector(taskLib[curId].errX, 
            //     taskLib[curId].taskName + ": errX");
            // CostomUtils::print_vector(taskLib[curId].kp * taskLib[curId].errX, 
            //         taskLib[curId].taskName + ": kd * errX");
            // std::cout << 
            //     std::endl << "=============END PriorityTask===========" << std::endl;
        } else {
            taskLib[curId].N = taskLib[parentId].N *
                    (Eigen::MatrixXd::Identity(taskLib[parentId].Jpre.cols(),taskLib
                        [parentId].Jpre.cols()) -
                    pseudoInv_right_weighted(taskLib[parentId].Jpre,taskLib[parentId].W) *
                    taskLib[parentId].Jpre);
            taskLib[curId].Jpre = taskLib[curId].J * taskLib[curId].N;
            taskLib[curId].delta_q =
                taskLib[parentId].delta_q +
                pseudoInv_right_weighted(taskLib[curId].Jpre,taskLib[curId].W) *
                (taskLib[curId].errX -
                    taskLib[curId].J * taskLib[parentId].delta_q);
            taskLib[curId].dq =
                taskLib[parentId].dq +
                pseudoInv_right_weighted(taskLib[curId].Jpre,taskLib[curId].W) *
                (taskLib[curId].dxDes -
                    taskLib[curId].J * taskLib[parentId].dq);
            Eigen::VectorXd ddxcmd =
                taskLib[curId].ddxDes +
                taskLib[curId].kp * taskLib[curId].errX +
                taskLib[curId].kd * taskLib[curId].derrX;
            taskLib[curId].ddq =
                taskLib[parentId].ddq +
                dyn_pseudoInv(taskLib[curId].Jpre,dyn_M_inv,true) *
                    (ddxcmd - taskLib[curId].dJ * dq -
                    taskLib[curId].J * taskLib[parentId].ddq);
        }

        if (childId!=-1) {
            parentId=curId;
            curId=childId;
            childId=taskLib[curId].childId;
        }
        else {
            break;
        }
    }
    out_delta_q = taskLib[curId].delta_q;
    out_dq = taskLib[curId].dq;
    out_ddq = taskLib[curId].ddq;
    // out_delta_q = Eigen::VectorXd::Zero(taskLib[curId].delta_q.size());
    // out_dq = Eigen::VectorXd::Zero(taskLib[curId].delta_q.size());
    // out_ddq = Eigen::VectorXd::Zero(taskLib[curId].delta_q.size());
    // std::cout << "*****kinWBC*****" << std::endl <<
    // out_delta_q.transpose() << std::endl <<
    // out_dq.transpose() << std::endl <<
    // out_ddq.transpose() << std::endl;
}


}














