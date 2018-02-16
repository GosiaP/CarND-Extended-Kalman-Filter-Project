#include "kalman_filter.h"
#include "tools.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;


KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) 
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateState(const Eigen::VectorXd& y)
{
  /**
  * update the state by using Kalman Filter equations
  */
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  UpdateState(z - H_ * x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */

    VectorXd zpred = Tools().ConvertCartesianToPolar(x_);
    VectorXd y = z - zpred;
    y[1] = Tools().NormalizeAngle(y[1]);
    UpdateState(y);
}
