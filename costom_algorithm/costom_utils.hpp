#pragma once
#include<iostream>
#include<iomanip>
#include<Eigen/Dense>

namespace CostomUtils {

template <typename Matrix>
void print_matrix(const Matrix& matrix, const std::string& name) {
    // 保存当前状态
    std::ios::fmtflags orig_flags = std::cout.flags();
    std::streamsize orig_precision = std::cout.precision();
    // 格式化输出矩阵
    std::cout << std::fixed << std::setprecision(2);
    Eigen::IOFormat matrix_clear_fmt(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    std::cout << std::endl << "----- " << name << " " << "(" << 
    matrix.rows() << "x" << matrix.cols() << ")" << "-----" << std::endl << 
    matrix.format(matrix_clear_fmt) << std::endl;
    // 恢复原有状态
    std::cout.flags(orig_flags);
    std::cout.precision(orig_precision);
}

template<typename Vector>
void print_vector(const Vector& vector, const std::string& name) {
    // 保存当前状态
    std::ios::fmtflags orig_flags = std::cout.flags();
    std::streamsize orig_precision = std::cout.precision();
    // 格式化输出向量
    std::cout << std::fixed << std::setprecision(2);
    Eigen::IOFormat vector_format(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    std::cout << name << ": " << 
    vector.transpose().format(vector_format) << std::endl;
    // 恢复原有状态
    std::cout.flags(orig_flags);
    std::cout.precision(orig_precision);
}

} // end namespace Utils
