#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

MatrixXd householder(MatrixXd A) {
    return A;
}

void qr_factorization(MatrixXd A, MatrixXd Q, MatrixXd R) {
    int m = A.rows();
    int n = A.cols();

    for (int i = 0; i < n; i++) {
        MatrixXd column = A(Eigen::seq(i, n - i), i);
        MatrixXd w = householder(column);
        MatrixXd I = Eigen::MatrixXd::Identity(m - i, m - i);
        MatrixXd H_hat = I - 2 * w * w;
    }
}

int main()
{
    MatrixXd A = MatrixXd::Random(3, 3);
    int m = A.rows();
    int n = A.cols();
    MatrixXd Q = Eigen::MatrixXd::Identity(m, m);
    MatrixXd R = Eigen::MatrixXd::Identity(m, m);
    std::cout << A << std::endl;
    qr_factorization(A, Q, R);
}


