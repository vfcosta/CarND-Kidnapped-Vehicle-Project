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

#include "particle_filter.h"

using namespace std;

// Engine for random distributions
random_device rd;
std::default_random_engine gen(rd());

// Add random gaussian noise to particle
Particle addNoise(Particle p, double std[]) {
	//std::default_random_engine gen;
	p.x = normal_distribution<double>(p.x, std[0])(gen);
	p.y = normal_distribution<double>(p.y, std[1])(gen);
	p.theta = normal_distribution<double>(p.theta, std[2])(gen);
	return p;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 300;

	// initialize particles and weights
	weights = vector<double>(num_particles);
	particles = vector<Particle>(num_particles);
	Particle gps;
	gps.x = x;
	gps.y = y;
	gps.theta = theta;
	for(int i=0; i<num_particles; i++) {
		Particle particle = addNoise(gps, std);
		particle.id = i;
		particle.weight = 1;
		particles[i] = particle;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double vy = velocity/yaw_rate;
	double ydt = delta_t * yaw_rate;
	double vdt = delta_t * velocity;
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];
		if(fabs(yaw_rate) > 1e-5) {
			p.x += vy * (sin(p.theta + ydt) - sin(p.theta));
			p.y += vy * (cos(p.theta) - cos(p.theta + ydt));
		} else {
			p.x += vdt * cos(p.theta);
			p.y += vdt * sin(p.theta);
		}
		p.theta += ydt;
		particles[i] = addNoise(p, std_pos);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		for(int j=0; j<predicted.size(); j++) {
			LandmarkObs prediction = predicted[j];
			double distance = dist(observations[i].x, observations[i].y, prediction.x, prediction.y);
			if (distance < min_dist) {
				observations[i].id = prediction.id;
				min_dist = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double w_ratio = 2*M_PI*std_landmark[0]*std_landmark[1];
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];

		// transform observations to map coordinate system
		for(int j=0; j<observations.size(); j++) {
			LandmarkObs observation = observations[j];
			LandmarkObs observation_transformed;
			observation_transformed.x = observation.x * cos(p.theta) - observation.y * sin(p.theta) + p.x;
			observation_transformed.y = observation.x * sin(p.theta) + observation.y * cos(p.theta) + p.y;
			observations[j] = observation_transformed;
		}

		// predicted map landmarks
		std::vector<LandmarkObs> predicted;
		for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
			if(dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
				LandmarkObs prediction;
				prediction.id = predicted.size();
				prediction.x = landmark.x_f;
				prediction.y = landmark.y_f;
				predicted.push_back(prediction);
			}
		}

		if (predicted.size()==0) {
			//cout << "no predictions" << endl;
			p.weight = 1e-20;
			weights[i] = p.weight;
			particles[i] = p;
			continue;
		}

		// associate obserations to predictions
		dataAssociation(predicted, observations);

		p.weight = 1;
		//cout << "SIZE O: " << observations.size() << endl;
		//cout << "SIZE P: " <<  predicted.size() << endl;
		for(int n=0; n<observations.size(); n++) {
			LandmarkObs observation = observations[n];
			//cout << n << " OBS: " << observation.id << " " <<  observation.x << endl;
			LandmarkObs prediction = predicted[observation.id];
			double dx = observation.x - prediction.x;
			double dy = observation.y - prediction.y;
			p.weight *= exp(-dx*dx/(2*std_landmark[0]*std_landmark[0]) - dy*dy/(2*std_landmark[1]*std_landmark[1]) ) / w_ratio;
			//cout << p.weight << " " <<  dx << " " << exp(-20) << " , " << prediction.x << endl;
		}
		weights[i] = p.weight;
		particles[i] = p;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

		default_random_engine gen;
		discrete_distribution<> dist(weights.begin(), weights.end());
		std::vector<Particle> new_particles(particles.size());
		for (int i=0; i<num_particles; i++) {
			new_particles[i] = particles[dist(gen)];
		}
		particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
