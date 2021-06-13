// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Update Niter at each iteration of RANSAC algorithm
int updateNiter(int currentNiter, int inlierNb, int sampleNb, int matchNb){
    int nextNiter;
    nextNiter = log(BETA)/log(1- pow( (float)inlierNb/(float)matchNb, sampleNb));
    if(nextNiter > currentNiter){
        nextNiter = currentNiter;
    }
    return nextNiter;
}

// Pick a sample of sampleNb matches randomly chosen in the dataset
vector<Match> pickSample (vector<Match>& matches, int sampleNb){
    const int matchSize = matches.size();
    if(sampleNb>matchSize){
        cerr << "At least 8 samples are necessary." << endl;
    }
    int samplesInd[sampleNb]; //array of the indexes of the matches in the sample
    vector<Match> sample;

    //generate random numbers
    for (int i=0;i<sampleNb;i++)
    {
        bool used;
        int n;
        do{
            n=rand()%matchSize;
            //check if number is already used:
            used=false;
            for (int j=0;j<i;j++)
                if (n == samplesInd[j]) //number is already used
                {
                    used=true;
                    break;
                }
        } while (used); //loop until new, unique number is found
        samplesInd[i] = n;
        sample.push_back(matches[n]);
    }
    return sample;
}

// Compute F with the 8-point algorithm
FMatrix<float,3,3> eightPointF (vector<Match>& sample){
    if(sample.size()<8){
        cerr << "Not enough matches received for F computation !" << endl;
    }
    // Normalization matrix
    FMatrix<float,3,3> Norm(0.f);
    Norm(0,0) = 0.001;
    Norm(1,1) = 0.001;
    Norm(2,2) = 1;
    FMatrix<float,9,9> A;
    for(int i=0; i<8; i++){
        DoublePoint3 p1;
        p1[0] = sample[i].x1; 
        p1[1] = sample[i].y1; 
        p1[2] = 1;

        DoublePoint3 p2;
        p2[0] = sample[i].x2; 
        p2[1] = sample[i].y2; 
        p2[2] = 1;

        // Normalization
        p1 = Norm*p1;
        p2 = Norm*p2;

        A(i,0) = p1[0]*p2[0];   A(i,1) = p1[0]*p2[1];   A(i,2) = p1[0];
        A(i,3) = p1[1]*p2[0];   A(i,4) = p1[1]*p2[1];   A(i,5) = p1[1];
        A(i,6) = p2[0];         A(i,7) = p2[1];         A(i,8) = 1;
    }
    for(int j=0; j<9; j++){
        A(8,j) = 0;
    }

    // Solve the linear system with svd
    FVector<float,9> S;
    FMatrix<float,9,9> U, V_T;
    svd(A,U,S,V_T);
    FMatrix<float,3,3> computedF;
    for (int k=0; k<3; k++){
        for(int l=0; l<3; l++){
            computedF(k,l)= V_T.getRow(8)[3*k+l];
        }
    }
    // Project orthogonally on rank2 set
    FVector<float,3> S2;
    FMatrix<float,3,3> U2, V_T2;
    svd(computedF,U2,S2,V_T2);
    S2[2] = 0;
    computedF = U2 * Diagonal(S2) * V_T2;

    // Normalization of computedF
    computedF = Norm*computedF*Norm;
    return computedF;
}

// Return the indexes of the inliers
vector<int> inliersInd (vector<Match>& matches, FMatrix<float,3,3>& tempF, float dist){
    vector<int> inliers;
    for(int i=0; i<matches.size(); i++){
        Match currentMatch = matches[i];
        DoublePoint3 point1;
        DoublePoint3 point2;
        point1[0] = currentMatch.x1; point1[1] = currentMatch.y1; point1[2] = 1;
        point2[0] = currentMatch.x2; point2[1] = currentMatch.y2; point2[2] = 1;
        FVector<float, 3> line;
        line = tempF*point2;
        float norm = sqrt(pow(line[0],2.0) + pow(line[1],2.0));
        line /= norm;
        if(abs(point1*line) < dist){
            inliers.push_back(i);
        }
    }
    return inliers;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    FMatrix<float,3,3> tempF;
    vector<int> inliers;
    vector<Match> sampleMatches;
    int samplingNb(0);
    int const sampleSize(8);
    while(samplingNb < Niter){
        samplingNb++;
        sampleMatches = pickSample(matches, sampleSize);
        tempF = eightPointF(sampleMatches);
        inliers = inliersInd(matches, tempF, distMax);
        if(inliers.size() > bestInliers.size()){
            cout << "There are now " << inliers.size() << "inliers at iteration " << samplingNb << endl;
            bestInliers = inliers;
            bestF = tempF;
            if(bestInliers.size()>60){ // avoids int overflow
                Niter = updateNiter(Niter, bestInliers.size(), sampleSize, matches.size());
            }
            cout << " We have N_iter = " << Niter  << endl;
        }
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        DoublePoint3 p1;
        DoublePoint3 p2;
        int w = I1.width();
        if(x>w){ // click on I2
            p2[0] = x - w; 
            p2[1] = y; 
            p2[2] = 1;
            FVector<float, 3> ep_line;
            ep_line = F*p2;
            drawLine(0,(-1)*ep_line[2]/ep_line[1],w,(-1)*(ep_line[2]+ep_line[0]*w)/ep_line[1], RED);
        }
        if(x<=w){ // click on I1
            p1[0] = x; 
            p1[1] = y; 
            p1[2] = 1;
            FVector<float, 3> ep_line;
            ep_line = transpose(F)*p1;
            drawLine(w, (-1)*(ep_line[2])/ep_line[1], 2*w, (-1)*(ep_line[2]+ep_line[0]*w)/ep_line[1], RED);
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
