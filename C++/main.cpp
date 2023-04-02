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

void qr_factorization(MatrixXd & A, MatrixXd & Q) {
    int m = A.rows();
    int n = A.cols();

    for (int i = 0; i < n; i++) {
        MatrixXd column = A(Eigen::seq(i, n - 1), i);
        VectorXd w = householder(column);
        MatrixXd I = Eigen::MatrixXd::Identity(m - i, m - i);

        MatrixXd H_hat = I - 2 * w * w.transpose();
        MatrixXd H(m, m);
        for (int r = 0; r < m;  r++) {
            for (int c = 0; c < n; c++) {
                if (r == c && r < i) {
                    H(r, c) = 1;
                }
                else if (r < i || c < i) {
                    H(r, c) = 0;
                }
                else {
                    H(r, c) = H_hat(r - i, c - i);
                }
            }
        }
        Q = Q * H;
        A = H * A;
    }
}

int main()
{
    MatrixXd A{
        {0, 3, 1},
        {0, 4, -2},
        {2, 1, 1},
    };
    MatrixXd rawA = A;

    Eigen::HouseholderQR<Eigen::MatrixXd> householderQR(A);

    int m = A.rows();
    int n = A.cols();
    MatrixXd Q = Eigen::MatrixXd::Identity(m, m);
    qr_factorization(A, Q);
    MatrixXd diff = rawA - Q * A;
    float error = diff.norm() / rawA.norm();
    std::cout << error << std::endl;
}


