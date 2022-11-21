#!/usr/bin/python3
 
import sympy
import numpy
import sympybotics
# 建立机器人模型
rbtdef = sympybotics.RobotDef("xMate7p",
            [(0,       0,      0.404, "q"),
             ("-pi/2", 0,          0, "q"),
             ("pi/2",  0,     0.4375, "q"),
             ("-pi/2", 0,          0, "q"),
             ("pi/2",  0,     0.4125, "q"),
             ("-pi/2", 0,          0, "q"),
             ("pi/2",  0,     0.2755, "q")],
            dh_convention="modified")   # # (alpha, a, d, theta)

# 设定重力加速度的值（沿z轴负方向）
rbtdef.gravityacc=sympy.Matrix([0.0, 0.0, -9.81])
# 设定摩擦力 库伦摩擦与粘滞摩擦
rbtdef.frictionmodel = {'Coulomb', 'viscous'}
rbtdef.driveinertiamodel = 'simplified'
# 显示动力学全参数
print(rbtdef.dynparms())
#构建机器人动力学模型
rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)
# 转换为C代码
tau_str = sympybotics.robotcodegen.robot_code_to_func('C', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)
print(tau_str) #打印
#计算并显示动力学模型的回归观测矩阵，转换为C代码
rbt.calc_base_parms()
rbt.dyn.baseparms
print(rbt.dyn.baseparms)# 打印最小参数集P
rbt.Hb_code
print(rbt.Hb_code)#打印观测矩阵
Yr = sympybotics.robotcodegen.robot_code_to_func('C', rbt.Hb_code, 'H', 'Hb_code', rbtdef)
print(Yr) #打印显示转换为C代码后的观测矩阵Yr
#把动力学全参数模型，关节力矩模型，观测矩阵和最小惯性参数集结果保存为txt
data=open("result.cpp",'w+')
print(rbt.dyn.dynparms,tau_str,Yr,rbt.dyn.baseparms,file=data)
data.close()
 