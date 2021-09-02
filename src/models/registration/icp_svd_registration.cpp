/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */

#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "glog/logging.h"

#include "lidar_localization/models/registration/icp_svd_registration.hpp"

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(
    const YAML::Node& node
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    // parse params:
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPSVDRegistration::ICPSVDRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPSVDRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    // set params:
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    euc_fitness_eps_ = euc_fitness_eps;
    max_iter_ = max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;
    input_target_kdtree_->setInputCloud(input_target_);

    return true;
}

bool ICPSVDRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);

    // init estimation:
    transformation_.setIdentity();
    
    //
    // TODO: first option -- implement all computing logic on your own
    //
    // do estimation:
    int curr_iter = 0;




    while (curr_iter < max_iter_) {
        // TODO: apply current estimation:
        CloudData::CLOUD_PTR current_input_source (new CloudData::CLOUD());
        pcl::transformPointCloud(*transformed_input_source, *current_input_source, transformation_);

        std::vector<Eigen::Vector3f> cor_m,cor_c;
        int cor_num = 0;

        // TODO: get correspondence:
        cor_num = GetCorrespondence(current_input_source, cor_m, cor_c);

        LOG(INFO)<<cor_num<<std::endl;

        // TODO: do not have enough correspondence -- break:
        if (cor_num < 10)
        {
            LOG(WARNING)<<"Do not have enough correspondence !" << std::endl;
            break;
        }
        // TODO: update current transform:
        Eigen::Matrix4f temp_transformation;
        GetTransform(cor_m, cor_c, temp_transformation);
        LOG(INFO)<<__LINE__<<std::endl;

        // TODO: whether the transformation update is significant:
        if(!IsSignificant(temp_transformation ,trans_eps_)) 
        {
            break;
        }
        LOG(INFO)<<__LINE__<<std::endl;

        // TODO: update transformation:
        transformation_ = temp_transformation * transformation_;

      //LOG(INFO) << transformation_(0,0) << " " << transformation_(0,1) << " "<< transformation_(0,2) << " "<< transformation_(0,3) << " "<< transformation_(1,0) 
       //                       << " "<< transformation_(1,1) << " " << transformation_(1,2) << " "<< transformation_(1,3) << " "<< transformation_(2,0) << " "<< transformation_(2,1) 
        //                      << " "<< transformation_(2,2) << " "<< transformation_(2,3) <<std::endl; 

        ++curr_iter;
    }

    // set output:
    result_pose = transformation_ * predict_pose;
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);
    
    return true;
}

size_t ICPSVDRegistration::GetCorrespondence(
    const CloudData::CLOUD_PTR &input_source, 
    std::vector<Eigen::Vector3f> &xs,
    std::vector<Eigen::Vector3f> &ys
) {
    const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;

    size_t num_corr = 0;
    for(int i =0; i< input_source->size(); i++)
    {
        std::vector<int> matches_k;
        std::vector<float> distances;
        //LOG(INFO)<<__LINE__<<std::endl;

        input_target_kdtree_->nearestKSearch(input_source->at(i),1 ,matches_k, distances);
        //LOG(INFO)<<__LINE__<<std::endl;

        if(distances[0] > MAX_CORR_DIST_SQR) continue;

        Eigen::Vector3f temp_s,temp_t;
    
        temp_t[0] = input_target_->at(matches_k[0]).x;
        temp_t[1] = input_target_->at(matches_k[0]).y;
        temp_t[2] = input_target_->at(matches_k[0]).z;

        temp_s[0] = input_source->at(i).x;
        temp_s[1] = input_source->at(i).y;
        temp_s[2] = input_source->at(i).z;

        xs.emplace_back(temp_t);
        ys.emplace_back(temp_s);
        num_corr++;
        //LOG(INFO)<<__LINE__<<std::endl;
    }

    // TODO: set up point correspondence

    return num_corr;
}

void ICPSVDRegistration::GetTransform(
    const std::vector<Eigen::Vector3f> &xs,
    const std::vector<Eigen::Vector3f> &ys,
    Eigen::Matrix4f &transformation_
) {
    const size_t N = xs.size();

    // TODO: find centroids of mu_x and mu_y:
    Eigen::Vector3f xa,ya;
    for(int i = 0; i < N; i++)
    {
        xa(0) += xs.at(i)(0);
        xa(1) += xs.at(i)(1);
        xa(2) += xs.at(i)(2);

        ya(0) += ys.at(i)(0);
        ya(1) += ys.at(i)(1);
        ya(2) += ys.at(i)(2);
    }
    xa(0) = xa(0)/N;
    xa(1) = xa(1)/N;
    xa(2) = xa(2)/N;
    ya(0) = ya(0)/N;
    ya(1) = ya(1)/N;
    ya(2) = ya(2)/N;

    LOG(INFO)<<xa(0)<<","<<xa(1)<<","<<xa(2)<<std::endl;
    LOG(INFO)<<ya(0)<<","<<ya(1)<<","<<ya(2)<<std::endl;

    std::vector<Eigen::Vector3f> xn,yn;
    Eigen::Vector3f x_temp,y_temp;

    for(int i = 0; i < N; i++)
    {
        x_temp = xs[i]-xa;
        y_temp = ys[i]-ya;
        xn.emplace_back(x_temp);
        yn.emplace_back(y_temp);
    }
    LOG(INFO)<<__LINE__<<std::endl;

    // TODO: build H:
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for(int i = 0; i < N; i++)
    {
        H += yn[i] * xn[i].transpose();
    }
    LOG(INFO)<<__LINE__<<std::endl;
    // TODO: solve R:
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R = V*U.transpose();
    // TODO: solve t:
    Eigen::Vector3f t = xa - R*ya;
    LOG(INFO) << t(0) <<"," << t(1) <<"," << t(2) << std::endl;
    Eigen::Quaternionf q(R);

    LOG(INFO) << q.x() <<"," << q.y() <<"," << q.z() << std::endl;

    // TODO: set output:
    transformation_.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    transformation_.block<3,1>(0,3) = t;
}

bool ICPSVDRegistration::IsSignificant(
    const Eigen::Matrix4f &transformation,
    const float trans_eps
) {
    // a. translation magnitude -- norm:
    float translation_magnitude = transformation.block<3, 1>(0, 3).norm();
    // b. rotation magnitude -- angle:
    float rotation_magnitude = fabs(
        acos(
            (transformation.block<3, 3>(0, 0).trace() - 1.0f) / 2.0f
        )
    );
    LOG(INFO)<<__LINE__<<std::endl;

    return (
        (translation_magnitude > trans_eps) || 
        (rotation_magnitude > trans_eps)
    );
}

} // namespace lidar_localization