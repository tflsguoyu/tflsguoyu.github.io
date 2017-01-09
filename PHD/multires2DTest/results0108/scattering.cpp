#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#include <ctime>
#include <iomanip>

#include <Eigen/Dense>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace Eigen;
//using namespace cv;
#define PI 3.1415926535897932384626433832795

int i, j;

/*
 * Thread-safe random number generator
 */

struct RNG {
    RNG() : distrb(0.0, 1.0), engines() {}

    void init(int nworkers) {
        std::random_device rd;
        engines.resize(nworkers);
        for ( int i = 0; i < nworkers; ++i )
            engines[i].seed(rd());
    }

    double operator()() {
        int id = omp_get_thread_num();
        return distrb(engines[id]);
    }

    std::uniform_real_distribution<double> distrb;
    std::vector<std::mt19937> engines;
} rng;

void getCoord(double x, double y, int H, int W, int &r, int &c) {
	
	r = ceil((1.0 - y)*H);
	c = ceil(x * W);
	r = r == 0 ? 1 : r; c = c == 0 ? 1 : c;
	r = r - 1; c = c - 1;
}

int main(int argc, char *argv[]) {
    int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);
	rng.init(nworkers);	

	ifstream file(argv[1]);
	const double albedo = atof(argv[2]);
	const int N_Sample = atoi(argv[3]);
	const int h_sigmaT_d = atoi(argv[4]);
	const int w_sigmaT_d = atoi(argv[5]);
	const double h = atof(argv[6]);
	const double w = atof(argv[7]);

	// read csv
	MatrixXd sigmaT_d_NN(h_sigmaT_d, w_sigmaT_d);
	for (int i = 0; i < h_sigmaT_d; ++i)
	{
		string line;
		getline(file, line);
		//if (!file.good())
		//	break;

		stringstream iss(line);
		for (int j = 0; j < w_sigmaT_d; ++j)
		{
			string val;
			getline(iss, val, ',');
			stringstream convertor(val);
			convertor >> sigmaT_d_NN(i,j);
		}
	}

	// compute MAX of sigmaT
	double sigmaT_MAX = sigmaT_d_NN.maxCoeff();

	// create density map 
	int h_mapSize = 32;
	int w_mapSize = h_mapSize * round(w / h);
	MatrixXd densityMapTotal = MatrixXd::Zero(h_mapSize, w_mapSize);
	vector<MatrixXd> densityMap(nworkers);
	for (i = 0; i < nworkers; ++i) 
		densityMap[i] = MatrixXd::Zero(h_mapSize, w_mapSize);

	// create output reflectance
	double reflectanceTotal = 0.0;
	VectorXd reflectance = VectorXd::Zero(nworkers);

	double reflectanceTotal2 = 0.0;
	VectorXd reflectance2 = VectorXd::Zero(nworkers);

	double reflectanceStderr = 0.0;

#pragma omp parallel for //schedule(dynamic, 1)
	for (int sample = 1; sample <= N_Sample; ++sample) {

		int tid = omp_get_thread_num();

		const int maxDepth = 1000;
		Vector2d pos(rng()*w, h);
		Vector2d dir(0.0, 1.0);

		int r, c;
		int row, col;
		double sigmaT, sigmaT_next;
		Vector2d pos_next;
		int r_next, c_next;

		double weight = 1.0;
		double refl = 0.0;

		for (int dep = 1; dep <= maxDepth; ++dep) {

			// Method 1: 
			//double t = -log(rng()) / sigmaT_MAX;

			// Method 2: Woodcock
			double t = 0.0;
			while (1) {
				t = t - log(rng()) / sigmaT_MAX;
				pos_next = pos - t * dir;
				if (pos_next(0) < 0 || pos_next(0) > w || pos_next(1) < 0 || pos_next(1) > h)
					break;
				getCoord(pos_next(0)/w, pos_next(1)/h, h_sigmaT_d, w_sigmaT_d, r_next, c_next);
				sigmaT_next = sigmaT_d_NN(r_next,c_next);
				if ((sigmaT_next / sigmaT_MAX) > rng())
					break;
			}
			
			pos = pos - t * dir;

			if (pos(0) < 0.0 || pos(0) > w || pos(1) < 0.0 || pos(1) > h)
				break;

			// 
			 /*Vector2d a(rng()*w, h);
			 double dis = sqrt((pos(0) - a(0)) * (pos(0) - a(0)) + (pos(1) - a(1)) * (pos(1) - a(1)));
			 double costheta = abs(pos(1) - a(1)) / dis;
			 double newWeight = exp(-dis * sigmaT_d_NN(10,10)) * (1.0 / (2.0 * PI)) * weight * w * (costheta / dis);
*/
			//
			Vector2d a(rng()*w, h);
			Vector2d dir_a = a - pos;
			double dis_a = dir_a.norm();
			dir_a = dir_a.normalized();
			double costheta = abs(pos(1) - a(1)) / dis_a;
			t = 0.0;
			while (1) {
				t = t - log(rng()) / sigmaT_MAX;
				pos_next = pos + t * dir_a;
				if (pos_next(1) > h)
					break;
				getCoord(pos_next(0) / w, pos_next(1) / h, h_sigmaT_d, w_sigmaT_d, r_next, c_next);
				sigmaT_next = sigmaT_d_NN(r_next, c_next);
				if ((sigmaT_next / sigmaT_MAX) > rng())
					break;
			}
			double newWeight = 0.0;
			if (t < dis_a)
				double newWeight = (1.0 / (2.0 * PI)) * weight * w * (costheta / dis_a);
			
			refl += newWeight;
			
			double dir_theta = 2.0 * PI * rng();
			dir << cos(dir_theta), sin(dir_theta);

			getCoord(pos(0) / w, pos(1) / h, h_sigmaT_d, w_sigmaT_d, r, c);
			getCoord(pos(0) / w, pos(1) / h, h_mapSize, w_mapSize, row, col);

			sigmaT = sigmaT_d_NN(r, c);
			densityMap[tid](row, col) += weight / sigmaT;

			//weight *= albedo;

			if (dep <= 10)
				weight *= albedo;
			else
				if (rng() > albedo) break;

		}
		reflectance(tid) += refl;
		reflectance2(tid) += refl * refl;


	}
	 
	for (i = 0; i < nworkers; ++i)
		reflectanceTotal += reflectance(i);
	reflectanceTotal = reflectanceTotal / N_Sample;
	
	for (i = 0; i < nworkers; ++i)
		reflectanceTotal2 += reflectance2(i);
	reflectanceTotal2 = reflectanceTotal2 / N_Sample;

	reflectanceStderr = sqrt(reflectanceTotal2 - reflectanceTotal * reflectanceTotal) / sqrt(N_Sample);

	for (i = 0; i < nworkers; ++i)
		densityMapTotal += densityMap[i];
	densityMapTotal = densityMapTotal / N_Sample;

	ofstream outfile;
	outfile.open("output/densityMap.csv");
	for (i = 0; i < h_mapSize; ++i)
	{
		outfile << fixed;
		outfile << setprecision(15) << densityMapTotal(i,0);

		for (j = 1; j < w_mapSize; ++j)
		{
			outfile << "," << setprecision(15) << densityMapTotal(i,j);
		}

		outfile << endl;
	}
	outfile.close();

	outfile.open("output/reflectance.csv");
	outfile << fixed;
	outfile << setprecision(15) << reflectanceTotal;
	outfile.close();

	outfile.open("output/reflectanceStderr.csv");
	outfile << fixed;
	outfile << setprecision(15) << reflectanceStderr;
	outfile.close();

	return 0;
}
