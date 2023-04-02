#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

VectorXd householder(VectorXd u) {
    float sigma = u.norm();
    int sign = u[0] >= 0 ? -1 : 1;
    VectorXd e = ArrayXd::Zero(u.size());
    e[0] = 1;
    VectorXd v = sign * sigma * e;
    u = u - v;
    VectorXd w = u / u.norm();
    return w;
}

void qr_factorization(MatrixXd A, MatrixXd Q, MatrixXd R) {
    int m = A.rows();
    int n = A.cols();

    for (int i = 0; i < n; i++) {
        MatrixXd column = A(Eigen::seq(i, n - 1), i);
        VectorXd w = householder(column);
        MatrixXd I = Eigen::MatrixXd::Identity(m - i, m - i);

        MatrixXd H_hat = I - 2 * w * w.transpose();

        std::cout << H_hat << std::endl;
        std::cout << H_hat << std::endl;
    }
}

int main()
{
    //MatrixXd A = MatrixXd::Random(3, 3);
    MatrixXd A{
        {0, 3, 1},
        {0, 4, -2},
        {2, 1, 1},
    };
    Eigen::HouseholderQR<Eigen::MatrixXd> householderQR(A);

    int m = A.rows();
    int n = A.cols();
    MatrixXd Q = Eigen::MatrixXd::Identity(m, m);
    MatrixXd R = Eigen::MatrixXd::Identity(m, m);
    std::cout << A << std::endl;
    qr_factorization(A, Q, R);
}


