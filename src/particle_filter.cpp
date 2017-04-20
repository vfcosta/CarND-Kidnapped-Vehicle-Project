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
std::default_random_engine gen;

// Add random gaussian noise to particle
Particle addNoise(Particle p, double std[]) {
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
	num_particles = 100;

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
		particles[i] = particle;
		weights[i] = 1;
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
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];
		p.x += vy * (sin(p.theta + ydt) - sin(p.theta));
		p.y += vy * (cos(p.theta) - cos(p.theta + ydt));
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
		LandmarkObs observation = observations[i];
		double min_dist = numeric_limits<double>::max();
		for(int j=0; j<predicted.size(); j++) {
			LandmarkObs prediction = predicted[i];
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			if (distance < min_dist) {
				observation.id = prediction.id;
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
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
