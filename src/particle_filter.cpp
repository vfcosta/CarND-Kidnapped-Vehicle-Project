#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

// initialize engine for random distributions
random_device rd;
std::default_random_engine gen(rd());

// Add random gaussian noise to particle
Particle addNoise(Particle p, double std[]) {
	p.x = normal_distribution<double>(p.x, std[0])(gen);
	p.y = normal_distribution<double>(p.y, std[1])(gen);
	p.theta = normal_distribution<double>(p.theta, std[2])(gen);
	return p;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 200;

	// initialize particles and weights
	weights = vector<double>(num_particles);
	particles = vector<Particle>(num_particles);
	Particle gps;
	gps.x = x;
	gps.y = y;
	gps.theta = theta;
	for(int i=0; i<num_particles; i++) {
		Particle particle = addNoise(gps, std); // Add random Gaussian noise to each particle
		particle.id = i;
		particle.weight = 1;
		particles[i] = particle;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// precalculate some values
	double vy = velocity/yaw_rate;
	double ydt = delta_t * yaw_rate;
	double vdt = delta_t * velocity;
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];
		// predict x,y for t+1
		if(fabs(yaw_rate) > 1e-5) {
			p.x += vy * (sin(p.theta + ydt) - sin(p.theta));
			p.y += vy * (cos(p.theta) - cos(p.theta + ydt));
		} else {
			p.x += vdt * cos(p.theta);
			p.y += vdt * sin(p.theta);
		}
		p.theta += ydt;
		particles[i] = addNoise(p, std_pos); // add noise to each particle
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// find landmark with minimal distance
	for(int i=0; i<observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		for(int j=0; j<predicted.size(); j++) {
			LandmarkObs prediction = predicted[j];
			double distance = dist(observations[i].x, observations[i].y, prediction.x, prediction.y);
			if (distance < min_dist) {
				observations[i].id = j;
				min_dist = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// precalculate denominator for multivariate gaussian
	double w_ratio = 2*M_PI*std_landmark[0]*std_landmark[1];
	for(int i=0; i<num_particles; i++) {
		Particle &p = particles[i];

		// transform observations to map coordinate system
		std::vector<LandmarkObs> observations_transformed;
		for(int j=0; j<observations.size(); j++) {
			LandmarkObs observation = observations[j];
			LandmarkObs observation_transformed;
			observation_transformed.x = observation.x * cos(p.theta) - observation.y * sin(p.theta) + p.x;
			observation_transformed.y = observation.x * sin(p.theta) + observation.y * cos(p.theta) + p.y;
			observations_transformed.push_back(observation_transformed);
		}

		// predicted map landmarks
		std::vector<LandmarkObs> predicted;
		for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
			if(dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
				LandmarkObs prediction;
				prediction.id = landmark.id_i;
				prediction.x = landmark.x_f;
				prediction.y = landmark.y_f;
				predicted.push_back(prediction);
			}
		}

		// skip further steps when predicted is empty
		if (predicted.size()==0) {
			p.weight = 1e-20;
			weights[i] = p.weight;
			continue;
		}

		// associate obserations to predictions
		dataAssociation(predicted, observations_transformed);

		p.weight = 1;
		for(int n=0; n<observations_transformed.size(); n++) {
			LandmarkObs observation = observations_transformed[n];
			LandmarkObs prediction = predicted[observation.id];
			double dx = observation.x - prediction.x;
			double dy = observation.y - prediction.y;
			p.weight *= exp(-dx*dx/(2*std_landmark[0]*std_landmark[0]) - dy*dy/(2*std_landmark[1]*std_landmark[1]) ) / w_ratio;
		}
		weights[i] = p.weight; // store weight in a vector to simplify resample
	}
}

void ParticleFilter::resample() {
		// Resample particles with replacement with probability proportional to their weight.
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
