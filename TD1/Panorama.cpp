// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    // Variables for mouse capture, p for position, win for the window, subwin for the suwindow and button (1, 2 or 3) for the button used
    IntPoint2 p;
    Window win;
    int subWin;
    int button = 1;

    cout << "Select the points with the left button of the mouse, and click with the right button or the roll button when you are done." << endl;

    // click with the right button or the roll of the mouse to exit
    while(button == 1){ 
        button = anyGetMouse(p,win,subWin);
        if(button == 1 && win == w1){
            pts1.push_back(p);
            // Graphical add-on to see the point and its number, from Clement's idea
            setActiveWindow(w1);
            drawString(p, to_string(pts1.size()), RED);
            drawCircle(p,3,RED,3,false);
        } else if(button == 1 && win == w2) {
            pts2.push_back(p);
            // Graphical add-on to see the point and its number, from Clement's idea
            setActiveWindow(w2);
            drawString(p, to_string(pts2.size()), RED);
            drawCircle(p,3,RED,3,false);
        }
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    // Construction of A and B
    for(int i=0;i<n;i++){
        A(2*i,0) = (double) pts1[i].x(); 
        A(2*i,1) = (double) pts1[i].y(); 
        A(2*i,2) = 1.; 
        A(2*i,3) = 0.; 
        A(2*i,4) = 0.; 
        A(2*i,5) = 0.; 
        A(2*i,6) = (double) -pts2[i].x()*pts1[i].x(); 
        A(2*i,7) = (double) -pts2[i].x()*pts1[i].y();

        A(2*i+1,0) = 0.; 
        A(2*i+1,1) = 0.; 
        A(2*i+1,2) = 0.; 
        A(2*i+1,3) = (double) pts1[i].x(); 
        A(2*i+1,4) = (double) pts1[i].y(); 
        A(2*i+1,5) = 1.; 
        A(2*i+1,6) = (double) -pts2[i].y()*pts1[i].x(); 
        A(2*i+1,7) = (double) -pts2[i].y()*pts1[i].y();

        B[2*i] = (double) pts2[i].x();
        B[2*i+1] = (double) pts2[i].y();
    }

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    cout << "Sanity check (no 0 value):" << endl;
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "Bounds of the panorama:" << endl;
    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);
    // ------------- TODO/A completer ----------
    // Iterate over the coordinates of the complete box 
    for(int i=int(x0);i<int(x0)+int(x1-x0);i++){
        for(int j=int(y0);j<int(y0)+int(y1-y0);j++){
            // Pull
            Vector<float> p(3);
            Vector<float> p_ini(3);
            // coordinates p_ini of (i,j) in the projective space
            p_ini[0] = (float) i;
            p_ini[1] = (float) j;
            p_ini[2] = (float) 1;
            // coordinates p = H^(-1)(p_ini), for the pull
            p = linSolve(H,p_ini);
            p /= p[2];
            
            // Default value for Dirichlet access
            RGB<unsigned char> b(0,0,0); 

            if(p[0]>=0 && p[0]<=I1.width()-1 && p[1]>=0 && p[1]<=I1.height()-1){
                if(i>=0 && i<=I2.width()-1 && j>=0 && j<=I2.height()-1){ 
                    // pixel in I1 and I2: averaging the intensities
                    // p[0] and p[1] are float, use of interpolation to find the values of the pixel in I1, and averaging with the value of the corresponding pixel in I2
                    I(Coords<2>(i-int(x0),j-int(y0))) = (RGB<unsigned char>) (I1.interpolate(p[0],p[1])/((float)2) + ((RGB<float>) I2.dirichlet(Coords<2> (i,j),b))/((float)2));
                } else { 
                    // pixel only in I1
                    // p[0] and p[1] are float, use of interpolation to find the values of the pixel in I1
                    I(Coords<2>(i-int(x0),j-int(y0))) = (RGB<unsigned char>) I1.interpolate(p[0],p[1]);
                }
            } else { 
                // pixel only in I2
                I(Coords<2>(i-int(x0),j-int(y0))) = I2.dirichlet(Coords<2> (i,j),b);
            }

        }
    }
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "The computed homography H:" << endl;
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
