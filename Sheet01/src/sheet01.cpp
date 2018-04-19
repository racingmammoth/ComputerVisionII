#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace std;
using namespace cv;

/** classification header **/
#define NUM_ITERATIONS 5
#define STEP_SIZE 1

struct ClassificationParam{
    string posTrain, negTrain;
    string posTest, negTest;
};

// regression class for various regression methods
class LogisticRegression{
private:
    Mat train, test;                // each column is a feature vector
    Mat gtLabelTrain, gtLabelTest;  // row vector
    Mat phi;

    int loadFeatures(std::string& trainFile, std::string& testFile, Mat& feat, Mat& gtLabel);

public:
    LogisticRegression(ClassificationParam& param);
    int learnClassifier(); // TODO implement
    int testClassifier(); // TODO implement
    float sigmoid(float a);
    ~LogisticRegression(){}
};

/** regression header **/
#define FIN_RBF_NUM_CLUST 300
#define RBF_SIGMA 1e-3

// reading input parameters
struct RegressionParam{
    std::string regressionTrain;
    std::string regressionTest;
};

// models for regression
class Model{
public:
    Mat phi;        // each row models wi
    Mat sigma_sq;   // column vector
    Mat codeBook;   // codebook for finite kernel reg.
};

// regression class for various regression methods
class Regression{
private:
    Mat   trainx, trainw;
    Mat   testx, testw;
    Model linear_reg, fin_rbf_reg, dual_reg;

    int loadFeatures(std::string& fileName, Mat& vecx, Mat& vecw);

public:
    Regression(RegressionParam& param);
    ~Regression(){}
    int trainLinearRegression(); // TODO implement
    int trainFinite_RBF_KernelRegression(); // TODO implement
    int trainDualRegression(); // TODO implement

    int testLinearRegresssion(); // TODO implement
    int testFinite_RBF_KernelRegression(); // TODO implement
    int testDualRegression(); // TODO implement

};

int main()
{
    RegressionParam rparam;
    rparam.regressionTrain = "../data/regression_train.txt";
    rparam.regressionTest  = "../data/regression_test.txt";

    Regression reg(rparam);

    // linear regression
    reg.trainLinearRegression();
    reg.testLinearRegresssion();
    //reg.trainFinite_RBF_KernelRegression();
    //reg.testFinite_RBF_KernelRegression();
    //reg.trainDualRegression();
    //reg.testDualRegression();

    ClassificationParam cparam;
    cparam.posTrain = "../data/bottle_train.txt";
    cparam.negTrain = "../data/horse_train.txt";
    cparam.posTest  = "../data/bottle_test.txt";
    cparam.negTest  = "../data/horse_test.txt";

    LogisticRegression cls(cparam);
    //cls.learnClassifier();
    //cls.testClassifier();
    cout << "Der Test läuft durch." << endl;

    return 0;
}

/** classification functions **/
LogisticRegression::LogisticRegression(ClassificationParam& param){

    loadFeatures(param.posTrain,param.negTrain,train,gtLabelTrain);
    loadFeatures(param.posTest,param.negTest,test,gtLabelTest);
}

int LogisticRegression::loadFeatures(string& trainPos, string& trainNeg, Mat& feat, Mat& gtL){

    ifstream iPos(trainPos.c_str());
    if(!iPos) {
        cout<<"error reading train file: "<<trainPos<<endl;
        exit(-1);
    }
    ifstream iNeg(trainNeg.c_str());
    if(!iNeg) {
        cout<<"error reading test file: "<<trainNeg<<endl;
        exit(-1);
    }

    int rPos, rNeg, cPos, cNeg;
    iPos >> rPos;
    iPos >> cPos;
    iNeg >> rNeg;
    iNeg  >> cNeg;

    if(cPos != cNeg){
        cout<<"Number of features in pos and neg classes unequal"<<endl;
        exit(-1);
    }
    feat.create(cPos+1,rPos+rNeg,CV_32F); // each column is a feat vect
    gtL.create(1,rPos+rNeg,CV_32F);       // row vector


    // load positive examples
    for(int idr=0; idr<rPos; ++idr){
        gtL.at<float>(0,idr) = 1;
        feat.at<float>(0,idr) = 1;
        for(int idc=0; idc<cPos; ++idc){
            iPos >> feat.at<float>(idc+1,idr);
        }
    }

    // load negative examples
    for(int idr=0; idr<rNeg; ++idr){
        gtL.at<float>(0,rPos+idr) = 0;
        feat.at<float>(0,rPos+idr) = 1;
        for(int idc=0; idc<cNeg; ++idc){
            iNeg >> feat.at<float>(idc+1,rPos+idr);
        }
    }

    iPos.close();
    iNeg.close();

    return 0;
}

float LogisticRegression::sigmoid(float a){
    return 1.0f/(1+exp(-a));
}

/** regression functions **/
Regression::Regression(RegressionParam& param){
    // load features
    loadFeatures(param.regressionTrain,trainx,trainw);
    loadFeatures(param.regressionTest,testx,testw);
//    cout<<"features loaded successfully"<<endl;

    // model memory
    linear_reg.phi.create(trainx.rows,trainw.rows,CV_32F); linear_reg.phi.setTo(0);
    linear_reg.sigma_sq.create(trainw.rows,1,CV_32F); linear_reg.sigma_sq.setTo(0);
    fin_rbf_reg.phi.create(FIN_RBF_NUM_CLUST,trainw.rows,CV_32F);
    fin_rbf_reg.sigma_sq.create(trainw.rows,1,CV_32F);
    dual_reg.phi.create(trainx.cols,trainw.rows,CV_32F);
    dual_reg.sigma_sq.create(trainw.rows,1,CV_32F);

}
int Regression::loadFeatures(string& fileName, Mat& matx, Mat& matw){

    // init dimensions and file
    int numR, numC, dimW;
    ifstream iStream(fileName.c_str());
    if(!iStream){
        cout<<"cannot read feature file: "<<fileName<<endl;
        exit(-1);
    }

    // read file contents
    iStream >> numR;
    iStream >> numC;
    iStream >> dimW;
    matx.create(numC-dimW+1,numR,CV_32F); // each column is a feature
    matw.create(dimW,numR,CV_32F);        // each column is a vector to be regressed

    for(int r=0; r<numR; ++r){
        // read world data
        for(int c=0; c<dimW; ++c)
            iStream >> matw.at<float>(c,r);
        // read feature data
        matx.at<float>(0,r)=1;
        for(int c=0; c<numC-dimW; ++c)
            iStream >> matx.at<float>(c+1,r);
    }
    iStream.close();

    return 0;
}
int Regression::trainLinearRegression() {
    Mat train_x = this->trainx;
    Mat train_x_transp;
    Mat train_w = this->trainw;
    Mat train_w_transp;
    transpose(train_w, train_w_transp);
    transpose(train_x, train_x_transp);
    Mat mult = train_x * train_x_transp;

    cout << train_x.rows << "   " << train_x.cols << endl;
    cout << train_x_transp.rows << "   " << train_x_transp.cols << endl;
    cout << mult.rows << "  " << mult.cols << endl;
    cout << train_w.rows << "   " << train_w.cols << endl;

    this->linear_reg.phi = (mult.inv() * train_x) * train_w_transp;
    Mat tmp = train_w_transp - train_x_transp * this->linear_reg.phi;
    Mat tmp_transp;
    transpose(tmp, tmp_transp);
    this->linear_reg.sigma_sq = (tmp_transp * tmp) / 603.;

    return 0;
}


int Regression::testLinearRegresssion() {
    cout << this->testw.rows << "  " << this->testw.cols << endl;
    cout << this->linear_reg.phi.rows << "  " << this->linear_reg.phi.cols << endl;
    Mat linear_reg_phi_transp;
    transpose(this->linear_reg.phi, linear_reg_phi_transp);
    cout << "IMPORTANTE" << endl;
    cout << this->testx.rows << "  " << this->testx.cols << endl;
    Mat inter2 = linear_reg_phi_transp * this->testx;
    cout << "inter2: " << inter2.rows << " " << inter2.cols << endl;
    cout << "Trivial" << endl;
    Mat inter = (this->testw - inter2);
    cout << "IMPORTANTE" << endl;
    cout << inter.rows << "  " << inter.cols << endl;

    Mat row1 = inter.row(0);
    Mat row2 = inter.row(1);
    Mat results(1, 100, CV_32F);
    double factor = 1./(std::sqrt(pow(2*CV_PI, 2)*determinant(this->linear_reg.sigma_sq)));

    for(int i = 0; i < 100; ++i){
        Mat tmp  = inter.col(i);
        cout << tmp.rows << " " << tmp.cols << endl;
        Mat tmp_transp;
        transpose(tmp, tmp_transp);
        cout << tmp_transp.rows << " " << tmp_transp.cols << endl;
        cout << this->linear_reg.sigma_sq.rows << endl;
        cout << -1./2*tmp_transp*(this->linear_reg.sigma_sq.inv())*tmp << endl;

        Mat exponential;
        exp(-1./2*tmp_transp*(this->linear_reg.sigma_sq.inv())*tmp, exponential);
        exponential *= factor;
        cout << exponential << endl;

        //double test = 1/(std::sqrt(2*CV_PI*determinant(this->linear_reg.sigma_sq)) * std::exp((-1/2*tmp_transp*(this->linear_reg.sigma_sq.inv())*tmp));
    }
    cout << "factor" << factor << endl;


    cout << row1.rows << " " << row1.cols << endl;
    cout << this->linear_reg.sigma_sq << endl;



    //cout << -1/2*row1*(this->linear_reg.sigma_sq.inv())*row1 << endl;
    //Mat results = 1/(std::sqrt(2*CV_PI*determinant(this->linear_reg.sigma_sq)) * gpu::exp(-1/2*row1*(this->linear_reg.sigma_sq.inv())*row1);
    return 0;
}


