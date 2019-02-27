/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>


#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 1000;
	
	double std_x;
	double std_y;
	double std_theta;	
	
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];	
	
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	for(int i= 0; i < num_particles; i++){
		Particle prt;
		prt.id = i;
		prt.x = dist_x(gen);
		prt.y = dist_y(gen);
		prt.theta = dist_theta(gen);
		prt.weight = 1;
		particles.push_back(prt);
		weights.push_back(1.0);
	}
	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
		
	double std_x;
	double std_y;
	double std_theta;	
	
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];		
		
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
		
	for(int i= 0; i < particles.size(); i++){
		double theta = particles[i].theta;
		if (abs(yaw_rate) > 0.0001) { 
			particles[i].x += (velocity/yaw_rate)*(sin(theta+yaw_rate*delta_t)-sin(theta))+dist_x(gen);
			particles[i].y += (velocity/yaw_rate)*(cos(theta) - cos(theta+yaw_rate*delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
		}else{
			particles[i].x += velocity*delta_t*(cos(theta)) + dist_x(gen);
			particles[i].y += velocity*delta_t*(sin(theta)) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}			
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for(int i = 0; i<observations.size(); i++){
		double t_dist = 100000;
		int new_id = 0;
		
		for(int j =0; j<predicted.size(); j++){
			double cur_dist;
			cur_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (cur_dist < t_dist){
				t_dist = cur_dist;
				new_id = predicted[j].id;
			}
		}
		observations[i].id = new_id;
	}
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
	
	for(int i = 0; i<particles.size(); i++){
		vector<LandmarkObs> t_obs(observations.size());
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		//transform to global CS
		for(int j = 0; j<observations.size(); j++){
			double xo = observations[j].x;
			double yo = observations[j].y;
			t_obs[j].x = x + xo*cos(theta) - yo*sin(theta);
			t_obs[j].y = y + xo*sin(theta) + yo*cos(theta);
			t_obs[j].id = observations[j].id;
		}
		
		vector<LandmarkObs> n_lm; //nearest Landmark
		
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];
			double xm = lm.x_f;
			double ym = lm.y_f;
			int idm = lm.id_i;
			
			double c_dist;
			c_dist = dist(x, y, xm, ym) ;
			
			if(c_dist<sensor_range){
				LandmarkObs pred_lm;
				pred_lm.id = idm;
				pred_lm.x = xm;
				pred_lm.y = ym;
				n_lm.push_back(pred_lm);
			}
			
		}
		
		dataAssociation(n_lm, t_obs);
		double t_weight;
		
		for(int j=0; j<t_obs.size(); j++){
			double x_obs = t_obs[j].x;
			double y_obs = t_obs[j].y;
			double mu_x;
			double mu_y;
			
			for (int k=0; k<n_lm.size(); k++){
				if(n_lm[k].id == t_obs[j].id){
					mu_x = n_lm[k].x;
					mu_y = n_lm[k].y;
				}
			}
			
			double exponent= pow((x_obs - mu_x),2)/(2 * sig_x*sig_x) + (pow((y_obs - mu_y),2)/(2 * sig_y*sig_y));
			t_weight= gauss_norm * exp(-exponent);
			
		}
		particles[i].weight = t_weight;
		weights[i] = t_weight;
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> d_dist(weights.begin(), weights.end());
	vector<Particle> t_particles;
	for(int i = 0; i <particles.size(); i++){
		int d_particle = d_dist(gen);
		t_particles.push_back(particles[d_particle]);
		weights[i] = particles[d_particle].weight;		
	}
	
	particles = t_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
