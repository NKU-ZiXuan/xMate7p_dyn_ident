#include <compute_torque.h>

#define ROBOT_DOF 7
#define PARMS_NUM 91
#define MIN_PARMS_NUM 62

int main(int argc, char** argv)
{
  // 关节状态
  static double q[ROBOT_DOF] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  static double dq[ROBOT_DOF] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  static double ddq[ROBOT_DOF] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  static double tau_out[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  
  
  // 第一种方法：全参数集计算力矩
  double parms[PARMS_NUM];
  // 读取全参数集
  std::string parms_file_path = "../src/V1_traj1_delta_starextra.dat";
  read_parms(parms, parms_file_path);
  
  // 计算全参数集辨识力矩
  tau(tau_out, parms, q, dq, ddq);
  
  for(int i=0; i<ROBOT_DOF; i++)
    std::cout << "joint " << i+1 << " toruqe: " << tau_out[i] << std::endl;

  // 第二种方法：最小惯性参数集计算力矩
  double parms_min[MIN_PARMS_NUM];
  // 读取最小惯性参数集
  std::string min_parms_file_path = "../src/V1_traj1_beta_starextra.dat";
  read_parms(parms_min, min_parms_file_path);

  static double H[ROBOT_DOF*MIN_PARMS_NUM];
  Hb_code(H, q, dq, ddq);
  // 回归矩阵
  MatrixXd W = Map<Matrix<double, ROBOT_DOF, MIN_PARMS_NUM, RowMajor>>(H);  
  // 最小惯性参数集
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> P(parms_min, MIN_PARMS_NUM, 1);
  // 计算力矩
  Matrix<double, ROBOT_DOF, 1> min_parms_tau_out;
  min_parms_tau_out = W * P;

  for(int i=0; i<ROBOT_DOF; i++)
    std::cout << "joint " << i+1 << " toruqe: " << min_parms_tau_out[i] << std::endl;
  return 0;
}