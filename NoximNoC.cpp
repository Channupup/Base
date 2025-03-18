/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the implementation of the Network-on-Chip
 */
#include "NoximNoC.h"
#include "NoximGlobalStats.h"
#include <sys/stat.h>
#include <fdeep/fdeep.hpp>
#include "mat.cpp"
using namespace std;
//the Variable for interlace
bool interlace_type;
int rema;
int best_grid[1][10][10];
std::vector<std::vector<std::vector<int>>> arr(DEFAULT_MESH_DIM_Z, std::vector<std::vector<int>>(DEFAULT_MESH_DIM_X, std::vector<int>(DEFAULT_MESH_DIM_Y)));
//the Variable for result
double sum_mse=0;
int mse_counter=0;
float MaxTempall;
//the Variable for prediction
const auto model = fdeep::load_model("LSTM_throt_93_level_80_start4_mpc0.json");
const auto model2 = fdeep::load_model("ANN_throt_93_level_80_start_len_10_1.json");

int beltway[20][20][4];		
// float temp_budget[20][20][4];	  // Derek 2012.10.16
// float thermal_factor[20][20][4];  // Derek 2012.12.14
// float penalty_factor[20][20][4];  // Derek 2012.12.17
// float MTTT[20][20][4];
int traffic[20][20][4];
int traffic2[20][20][4];
//int num_pkt = 2800;
// double consumption_rate[20][20][4];

int throttling2[20][20][4];
bool throttling[20][20][4];
float deltaT[20][20][4];
extern ofstream transient_log_throughput;
extern ofstream transient_topology;
extern ofstream traffic_analysis;
extern ofstream traffic_period;  //目前没有用这个变量
extern ofstream pretemp_file;
extern ofstream throt_analysis;
extern ofstream throt_level;
//the function for result
int NoximNoC::randInt(int min, int max)
{
    return min +
	(int) ((double) (max - min + 1) * rand() / (RAND_MAX + 1.0));
}
void NoximNoC::printmse(){
	cout<<"MSE = "<<sum_mse/mse_counter<<endl;
}
//the function for interlace
std::vector<std::vector<std::vector<int>>> read_3d_array_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件：" << filename << std::endl;
        return {};
    }

    std::vector<std::vector<std::vector<int>>> result;
    std::string line;
    std::vector<std::vector<int>> currentSlice;
    while (std::getline(file, line)) {
        if (line.find("--- Slice ") != std::string::npos) {
            // 遇到新的切片标记，将当前切片添加到结果中并重置
            if (!currentSlice.empty()) {
                result.push_back(currentSlice);
                currentSlice.clear();
            }
            continue;
        }
        if (line.empty()) {
            // 空行也视为新切片的开始或结束，这里简化处理直接跳过
            continue;
        }
        std::istringstream iss(line);
        std::vector<int> row;
        std::string number;
        while (std::getline(iss, number, ',')) {
            row.push_back(std::stoi(number));
        }
        currentSlice.push_back(row);
    }
    // 不要忘记添加最后一个切片
    if (!currentSlice.empty()) {
        result.push_back(currentSlice);
    }
    file.close();
    return result;
}
//the function for prediction
double NoximNoC::LSTMpredictor(int level,double x1,double x2,double x3,double x4, double x5, double x6, double x7,double x8, double x9, double x10){
	fdeep::tensor ten(fdeep::tensor_shape(10, 2), 0);
		ten.set(fdeep::tensor_pos( 0, 0), x10);
		ten.set(fdeep::tensor_pos( 1, 0), x9);
		ten.set(fdeep::tensor_pos( 2, 0), x8);
		ten.set(fdeep::tensor_pos( 3, 0), x7);
		ten.set(fdeep::tensor_pos( 4, 0), x6);
		ten.set(fdeep::tensor_pos( 5, 0), x5);
		ten.set(fdeep::tensor_pos( 6, 0), x4);
		ten.set(fdeep::tensor_pos( 7, 0), x3);
		ten.set(fdeep::tensor_pos( 8, 0), x2);
		ten.set(fdeep::tensor_pos( 9, 0), x1);
		for(int j=0; j<9; j++){
			ten.set(fdeep::tensor_pos(j, 1), 0);
		}
		ten.set(fdeep::tensor_pos( 9, 1), level);
		std::vector<fdeep::internal::tensor>  input;
		input.push_back(ten);
		const auto result = model.predict(input);
		const auto result_vec = result.front().to_vector();
		double out_float = result_vec.front()*20+80;
		return out_float;
}

void NoximNoC::predictor(vector<int> &throt_k,vector<double> &T_pre,double x1,double x2,double x3,double x4, double x5, double x6, double x7,double x8, double x9, double x10){
	// generate new temperature prediction sequence based on control sequence 
	// cout<<"generate new temperature prediction sequence based on control sequence "<<endl;
	for(int i=0;i<NoximGlobalParams::steplen;i++){
		// cout<<x1<<"--"<<x2<<"--"<<x3<<"--"<<x4<<"--"<<x5<<"--"<<x6<<"--"<<
		// x7<<"--"<<x8<<"--"<<x9<<"--"<<x10<<"--"<<throt_k[i]<<"--"<<endl;
		T_pre[i]=LSTMpredictor(throt_k[i],x1,x2,x3,x4,x5,x6,x7,x8,x9,x10);
		x10=x9;
		x9=x8;
		x8=x7;
		x7=x6;
		x6=x5;
		x5=x4;
		x4=x3;
		x3=x2;	
		x2=x1;
		x1=(T_pre[i]-80)/20;	
	}
	bool display=false;
	if(display){
	vector<int>::iterator it = throt_k.begin();
	while(it<throt_k.end()){
		cout<<*it<<"----";
		it++;
	}
	cout<<endl;
	
	vector<double>::iterator it2 = T_pre.begin();
	while(it2<T_pre.end()){
		cout<<*it2<<"----";
		it2++;
	}
	cout<<endl;}

}

double NoximNoC::cost_function(int m, int n, int o, vector<int> &throt_k, vector<double> &T_pre){
		//calculate the cost value of the control sequence an the corresponding temperature prediction sequence 
		double cost_value=0;
		for(int i=0;i<NoximGlobalParams::steplen;i++){
			if(i==0)
				cost_value+=pow(0.1,i)*(pow((T_pre[i]-NoximGlobalParams::threshold_para),2)+0.01*abs(throt_k[i]-t[m][n][o]->r->stats.throt_d));
			else
				cost_value+=pow(0.1,i)*(pow((T_pre[i]-NoximGlobalParams::threshold_para),2)+0.01*abs(throt_k[i]-throt_k[i-1]));
		}
//cout<<" calculate the cost value of the control sequence an the corresponding temperature prediction sequence "<<endl;
		bool display=false;
		if(display){
		cout<<"cost_value="<<cost_value<<endl;}
		

		return cost_value;
}

bool NoximNoC::Optimizer(vector<int> &throt_k, vector<double> &T_pre,double x1,double x2,double x3,double x4, double x5, double x6, double x7,double x8, double x9, double x10){
	//cout<<" generate new control sequence "<<endl;
	bool U_flag=false;
	U_flag=(throt_k[0]==3)||(T_pre[0]<NoximGlobalParams::threshold_para);
	for(int i=1;i<NoximGlobalParams::steplen;i++){
		U_flag= U_flag && ((throt_k[i]==3)||(T_pre[i]<NoximGlobalParams::threshold_para));
	}
	if(!U_flag){
		for(int i=0;i<NoximGlobalParams::steplen;i++){
		if(T_pre[i]>NoximGlobalParams::threshold_para)
			if(throt_k[i]<3){
				throt_k[i]+=1;
				break;
			}

		}
		predictor(throt_k,T_pre,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10);
	}
	

	U_flag=(throt_k[0]==3)||(T_pre[0]<NoximGlobalParams::threshold_para);
	for(int i=1;i<NoximGlobalParams::steplen;i++){
		U_flag= U_flag && ((throt_k[i]==3)||(T_pre[i]<NoximGlobalParams::threshold_para));
	}

	
	// vector<int>::iterator it = throt_k.begin();
	// while(it<throt_k.end()){
	// 	cout<<*it<<"----";
	// 	it++;
	// }
	// cout<<endl;

	return U_flag;
}


double NoximNoC::tmp_predict(int m, int n, int o, double x1,double x2,double x3,double x4, double x5, double x6, double x7,double x8, double x9, double x10){
	if(NoximGlobalParams::pre==1){
	bool U_flag=false;
	double cost=0;
	int count=1;
	double temp_cost=0;
	bool initial_state_flag=true;
	vector<int> throt_steplen_buffer;
	vector<double> pretemp_steplen_buffer;
	for(int i=0;i<NoximGlobalParams::steplen;i++){
		throt_steplen_buffer.push_back(0);
		pretemp_steplen_buffer.push_back(0);
	}
	predictor(throt_steplen_buffer,pretemp_steplen_buffer,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10);
	cost = cost_function(m,n,o,throt_steplen_buffer,pretemp_steplen_buffer);
	int uk=throt_steplen_buffer[0];
	while(!U_flag){
		
		U_flag=Optimizer(throt_steplen_buffer,pretemp_steplen_buffer,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10);
		for(int i=0;i<NoximGlobalParams::steplen;i++){
			initial_state_flag=initial_state_flag&&(throt_steplen_buffer[i]==0);
		}
		if(!(initial_state_flag&&U_flag))
			count++;
		temp_cost=cost_function(m,n,o,throt_steplen_buffer,pretemp_steplen_buffer);
		if(temp_cost<cost){
			cost=temp_cost;
			uk=throt_steplen_buffer[0];
		}
	}
	t[m][n][o]->r->stats.throt_d = uk;
	
	bool display=false;
	if(display){
	vector<int>::iterator it = throt_steplen_buffer.begin();
	while(it<throt_steplen_buffer.end()){
		cout<<*it<<"----";
		it++;
	}
	cout<<endl;
	cout<<"count:"<<count<<endl;
	//cout<<"uk:"<<uk<<endl;
	cout<<"******************************************end of contol*********************************************"<<endl;}
	return pretemp_steplen_buffer[0];

	
	}
	else if(NoximGlobalParams::pre==2){
		//ARMA pre  include mat.cpp
	int p = 1;
	int k=11;
	double out_float;
    	vector<Double> data,a;
    	vector<vector<Double> > y; //y = [data[p+1, ..., n]]
	vector<vector<Double> > x;
    	
    	data.push_back(x10*20+85);data.push_back(x9*20+80);data.push_back(x8*20+80);data.push_back(x7*20+80);data.push_back(x6*20+80);
	data.push_back(x5*20+80);data.push_back(x4*20+80);data.push_back(x3*20+80);data.push_back(x2*20+80);data.push_back(x1*20+80);
	
	//LeastSquares
	//a = LeastSquares(data,p);	
	vector<Double> tmpy;
	for(int i=p;i<data.size();i++){
		tmpy.push_back(data[i]);
		vector<Double> tmp;
		for(int j=i-1;j>=i-p;j--){
			tmp.push_back(data[j]);
		}
		tmp.push_back(1);
		x.push_back(tmp);
		//add a vector 0f 1
	}
	y.push_back(tmpy);
	y = tM(y);
	vector<vector<Double> > a2, tx,invx;
	tx = tM(x);
	invx = inv(mulMat(tx, x));
	a2 = mulMat(mulMat(invx,tx), y);
	a2 = tM(a2);
	a=a2[0];
	
	
	//predict
	//x = predict(data,a,11,p);
	Double res;
	for(int i=data.size();i<k;i++){
		Double s = 0;
		int SZ = data.size();
		for(int j=0;j<p;j++){
			s += a[j] * data[SZ-j-1];
		}
		s+=a[1];
		data.push_back(s);
	}
	out_float=data[k-1];
	if( out_float> NoximGlobalParams::threshold_para){
			//isEmergency = true;
			double diff;
			diff = out_float -  NoximGlobalParams::threshold_para;
			if(diff<0.2)
				t[m][n][o]->r->stats.throt_d = 1;
			else if(diff<0.7)
				t[m][n][o]->r->stats.throt_d = 2;
			else
				t[m][n][o]->r->stats.throt_d = 3;
	}
	else{ t[m][n][o]->r->stats.throt_d = 0; }

	return out_float;
	}
	else if(NoximGlobalParams::pre==3){

		double Thro_above;
		double Thro_below;
		double Thro_right;
		double Thro_left;
		double Thro_down;
		double Thro_up;
		double Thro_itself;
		double Temp_above;
		double Temp_below;
		double Temp_right;
		double Temp_left;
		double Temp_down;
		double Temp_up;
		double Temp_itself;

		if(o==0){
			Thro_above=0;
			Thro_below=throttling2[m][n][o+1];
			Temp_above=0;
			Temp_below=TemperatureTrace[3*(xyz2Id( m, n, o+1))]-80;
		}
		else if(o==2){
			Thro_above=throttling2[m][n][o-1];
			Thro_below=0;
			Temp_above=TemperatureTrace[3*(xyz2Id( m, n, o-1))]-80;
			Temp_below=0;
		}
		else{
			Thro_above=throttling2[m][n][o-1];
			Thro_below=throttling2[m][n][o+1];
			Temp_above=TemperatureTrace[3*(xyz2Id( m, n, o-1))]-80;
			Temp_below=TemperatureTrace[3*(xyz2Id( m, n, o+1))]-80;
		}

		if(m==0){
			Thro_right=throttling2[m+1][n][o];
			Thro_left=0;
			Temp_right=TemperatureTrace[3*(xyz2Id( m+1, n, o))]-80;
			Temp_left=0;
		}
		else if(m==7){
			Thro_right=0;
			Thro_left=throttling2[m-1][n][o];
			Temp_right=0;
			Temp_left=TemperatureTrace[3*(xyz2Id( m-1, n, o))]-80;
		}
		else{
			Thro_right=throttling2[m+1][n][o];
			Thro_left=throttling2[m-1][n][o];
			Temp_right=TemperatureTrace[3*(xyz2Id( m+1, n, o))]-80;
			Temp_left=TemperatureTrace[3*(xyz2Id( m-1, n, o))]-80;
		}

		if(n==0){
			Thro_down=0;
			Thro_up=throttling2[m][n+1][o];
			Temp_down=0;
			Temp_up=TemperatureTrace[3*(xyz2Id( m, n+1, o))]-80;
		}
		else if(n==7){
			Thro_down=throttling2[m][n-1][o];
			Thro_up=0;
			Temp_down=TemperatureTrace[3*(xyz2Id( m, n-1, o))]-80;
			Temp_up=0;
		}
		else{
			Thro_down=throttling2[m][n-1][o];
			Thro_up=throttling2[m][n+1][o];
			Temp_down=TemperatureTrace[3*(xyz2Id( m, n-1, o))]-80;
			Temp_up=TemperatureTrace[3*(xyz2Id( m, n+1, o))]-80;
		}
		
		Thro_itself=throttling2[m][n][o];
		Temp_itself=TemperatureTrace[3*(xyz2Id( m, n, o))]-80;

		//const auto model = fdeep::load_model("fdeep_model.json");
		fdeep::tensor ten(fdeep::tensor_shape(14), 0);
		ten.set(fdeep::tensor_pos( 0), Thro_above/20);
		ten.set(fdeep::tensor_pos( 1), Thro_below/20);
		ten.set(fdeep::tensor_pos( 2), Thro_right/20);
		ten.set(fdeep::tensor_pos( 3), Thro_left/20);
		ten.set(fdeep::tensor_pos( 4), Thro_down/20);
		ten.set(fdeep::tensor_pos( 5), Thro_up/20);
		ten.set(fdeep::tensor_pos( 6), Thro_itself/20);
		ten.set(fdeep::tensor_pos( 7), Temp_above/20);
		ten.set(fdeep::tensor_pos( 8), Temp_below/20);
		ten.set(fdeep::tensor_pos( 9), Temp_right/20);
		ten.set(fdeep::tensor_pos(10), Temp_left/20);
		ten.set(fdeep::tensor_pos(11), Temp_down/20);
		ten.set(fdeep::tensor_pos(12), Temp_up/20);
		ten.set(fdeep::tensor_pos(13), Temp_itself/20);

		
		
		std::vector<fdeep::internal::tensor>  input;
		input.push_back(ten);

		const auto result = model2.predict(input);
		const auto result_vec = result.front().to_vector();
		double out_float = result_vec.front()*20+80;

		cout<<"("<<m<<")"<<"("<<n<<")"<<"("<<o<<")"<<Thro_above/20<<"---"<<Thro_below/20<<"---"<<Thro_right/20<<"---"<<
		Thro_left/20<<"---"<<Thro_down/20<<"---"<<Thro_up/20<<"---"<<Thro_itself/20<<"---"<<Temp_above/20<<"---"<<
		Temp_below/20<<"---"<<Temp_right/20<<"---"<<Temp_left/20<<"---"<<Temp_down/20<<"---"<<Temp_up/20<<"---"<<
		Temp_itself/20<<"---"<<out_float<<endl;

		if( out_float> NoximGlobalParams::threshold_para){
			//isEmergency = true;
			double diff;
			diff = out_float -  NoximGlobalParams::threshold_para;
			if(diff<0.2)
				t[m][n][o]->r->stats.throt_d = 1;
			else if(diff<0.7)
				t[m][n][o]->r->stats.throt_d = 2;
			else
				t[m][n][o]->r->stats.throt_d = 3;
		}
		else{ t[m][n][o]->r->stats.throt_d = 0; }

		return out_float;
	}
}

void NoximNoC::LSTMpre(){
	int m, n, o;
	int idx = 0;
	double in1 = 0;
	double in2 = 0;	
	double in3 = 0;
	double in4 = 0;
	double in5 = 0;
	double in6 = 0;
	double in7 = 0;
	double in8 = 0;
	double in9 = 0;
	double in10 = 0;
	double pre_temperature=0;
	double error = 0;
	cout<<" enter "<<endl;
	
	pretemp_file<<"Cycletime: "<<getCurrentCycleNum()<<"\n";


	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++){
	pretemp_file<<"XY"<<o<<"=[\n";
	for(n=NoximGlobalParams::mesh_dim_y-1; n > -1; n--){
	for(m=0; m < NoximGlobalParams::mesh_dim_x; m++){
		idx = xyz2Id( m, n, o);
		t[m][n][o]->r->stats.last_temperature9 = t[m][n][o]->r->stats.last_temperature8;
		t[m][n][o]->r->stats.last_temperature8 = t[m][n][o]->r->stats.last_temperature7;
		t[m][n][o]->r->stats.last_temperature7 = t[m][n][o]->r->stats.last_temperature6;
		t[m][n][o]->r->stats.last_temperature6 = t[m][n][o]->r->stats.last_temperature5;
		t[m][n][o]->r->stats.last_temperature5 = t[m][n][o]->r->stats.last_temperature4;
		t[m][n][o]->r->stats.last_temperature4 = t[m][n][o]->r->stats.last_temperature3;
		t[m][n][o]->r->stats.last_temperature3 = t[m][n][o]->r->stats.last_temperature2;
		t[m][n][o]->r->stats.last_temperature2 = t[m][n][o]->r->stats.last_temperature1;
		t[m][n][o]->r->stats.last_temperature1 = t[m][n][o]->r->stats.temperature;
		t[m][n][o]->r->stats.temperature      = TemperatureTrace[3*idx];
		if(getCurrentCycleNum()>=1000000){
			in1 = (t[m][n][o]->r->stats.temperature-80)/20;
			in2 = (t[m][n][o]->r->stats.last_temperature1-80)/20;
			in3 = (t[m][n][o]->r->stats.last_temperature2-80)/20;
			in4 = (t[m][n][o]->r->stats.last_temperature3-80)/20;
			in5 = (t[m][n][o]->r->stats.last_temperature4-80)/20;
			in6 = (t[m][n][o]->r->stats.last_temperature5-80)/20;
			in7 = (t[m][n][o]->r->stats.last_temperature6-80)/20;
			in8 = (t[m][n][o]->r->stats.last_temperature7-80)/20;
			in9 = (t[m][n][o]->r->stats.last_temperature8-80)/20;
			in10 = (t[m][n][o]->r->stats.last_temperature9-80)/20;
			pre_temperature = tmp_predict(m,n,o,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10);
			t[m][n][o]->r->stats.last_pre_temperature7 = t[m][n][o]->r->stats.last_pre_temperature6;
			t[m][n][o]->r->stats.last_pre_temperature6 = t[m][n][o]->r->stats.last_pre_temperature5;
			t[m][n][o]->r->stats.last_pre_temperature5 = t[m][n][o]->r->stats.last_pre_temperature4;
			t[m][n][o]->r->stats.last_pre_temperature4 = t[m][n][o]->r->stats.last_pre_temperature3;
			t[m][n][o]->r->stats.last_pre_temperature3 = t[m][n][o]->r->stats.last_pre_temperature2;
			t[m][n][o]->r->stats.last_pre_temperature2 = t[m][n][o]->r->stats.last_pre_temperature1;
			t[m][n][o]->r->stats.last_pre_temperature1 = pre_temperature;
			pretemp_file<< pre_temperature << "\t";
			//dout<< TemperatureTrace[3*idx] << "\t";
		}
		else{
			pretemp_file<<TemperatureTrace[3*idx]<<"\t";
		}

		if(getCurrentCycleNum()>=1100000){
			error = error+pow((t[m][n][o]->r->stats.last_pre_temperature2-t[m][n][o]->r->stats.temperature),2);
		}
	}
	pretemp_file<<"\n";
	}
	pretemp_file<<"]\n"<<"\n";
	}
	pretemp_file.flush();
	cout<<in1<<"--"<<in2<<"--"<<in3<<"--"<<in4<<"--"<<in5<<"--"<<in6<<"--"<<in7<<"--"<<in8<<"--"<<in9<<"--"<<in10<<"--PRE:"<<pre_temperature<<"--"<<endl;
	cout<<(error/256)<<endl;
	if(getCurrentCycleNum()>=1100000){
	sum_mse = sum_mse + error/256;
	mse_counter = mse_counter+1;
	cout<<"MSE:  "<< (sum_mse/mse_counter)<<endl;
	}
	if(getCurrentCycleNum()==7900000){
	transient_log_throughput<<" MSE:  "<< (sum_mse/mse_counter)<<endl;
	}
}

void NoximNoC::buildMesh()
{
	cout<<"Start buildMesh..."<<endl;
    // Check for routing table availability
    if (NoximGlobalParams::routing_algorithm == ROUTING_TABLE_BASED)
	assert(grtable.load(NoximGlobalParams::routing_table_filename));

    // Check for traffic table availability
    if (NoximGlobalParams::traffic_distribution == TRAFFIC_TABLE_BASED)
	assert(gttable.load(NoximGlobalParams::traffic_table_filename));
	
    // Create the mesh as a matrix of tiles
	int i,j,k;
	char _name[40];
	for ( i = 0; i < NoximGlobalParams::mesh_dim_x; i++) 
	for ( j = 0; j < NoximGlobalParams::mesh_dim_y; j++){
		sprintf( _name, "VLink[%02d][%02d]", i, j);
		v[i][j] = new NoximVLink( _name );
		v[i][j]->clock(clock);
		v[i][j]->reset(reset);
		v[i][j]->setId( i + NoximGlobalParams::mesh_dim_x*j );
		for ( k = 0; k < NoximGlobalParams::mesh_dim_z; k++){
			// Create the single Tile with a proper name
			sprintf(_name, "Tile[%02d][%02d][%02d]", i, j, k);
			t[i][j][k] = new NoximTile(_name);
			// Tell to the router its coordinates
			t[i][j][k]->r->configure( xyz2Id( i , j , k ), NoximGlobalParams::stats_warm_up_time,
					NoximGlobalParams::buffer_depth, grtable);
			//cout<<" set NoC ..."<<endl;
			// Tell to the PE its coordinates
			t[i][j][k]->pe->local_id       = xyz2Id( i , j , k );
			t[i][j][k]->pe->traffic_table  = &gttable;	// Needed to choose destination
			t[i][j][k]->pe->never_transmit = (gttable.occurrencesAsSource(t[i][j][k]->pe->local_id) == 0);
			//cout<<"get set NoC ..."<<endl;
			// Map clock and reset
			t[i][j][k]->clock(clock);
			t[i][j][k]->reset(reset);
			
			// Map Rx signals                                                
			t[i][j][k]->req_rx             [DIRECTION_NORTH] (req_to_south       [i  ][j  ][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_NORTH] (flit_to_south      [i  ][j  ][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_NORTH] (ack_to_north       [i  ][j  ][k  ]);
																							
			t[i][j][k]->req_rx             [DIRECTION_EAST ] (req_to_west        [i+1][j  ][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_EAST ] (flit_to_west       [i+1][j  ][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_EAST ] (ack_to_east        [i+1][j  ][k  ]);
																							
			t[i][j][k]->req_rx             [DIRECTION_SOUTH] (req_to_north       [i  ][j+1][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_SOUTH] (flit_to_north      [i  ][j+1][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_SOUTH] (ack_to_south       [i  ][j+1][k  ]);
																				
			t[i][j][k]->req_rx             [DIRECTION_WEST ] (req_to_east        [i  ][j  ][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_WEST ] (flit_to_east       [i  ][j  ][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_WEST ] (ack_to_west        [i  ][j  ][k  ]);
			
			//To VLink																	
			t[i][j][k]->req_rx             [DIRECTION_UP   ] (req_toT_down       [i  ][j  ][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_UP   ] (flit_toT_down      [i  ][j  ][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_UP   ] (ack_toV_up         [i  ][j  ][k  ]);
			//To VLink																			
			t[i][j][k]->req_rx             [DIRECTION_DOWN ] (req_toT_up         [i  ][j  ][k  ]);
			t[i][j][k]->flit_rx            [DIRECTION_DOWN ] (flit_toT_up        [i  ][j  ][k  ]);
			t[i][j][k]->ack_rx             [DIRECTION_DOWN ] (ack_toV_down       [i  ][j  ][k  ]);
			
			// Map Tx signals                                                    
			t[i][j][k]->req_tx             [DIRECTION_NORTH] (req_to_north       [i  ][j  ][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_NORTH] (flit_to_north      [i  ][j  ][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_NORTH] (ack_to_south       [i  ][j  ][k  ]);
																				
			t[i][j][k]->req_tx             [DIRECTION_EAST ] (req_to_east        [i+1][j  ][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_EAST ] (flit_to_east       [i+1][j  ][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_EAST ] (ack_to_west        [i+1][j  ][k  ]);
																				
			t[i][j][k]->req_tx             [DIRECTION_SOUTH] (req_to_south       [i  ][j+1][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_SOUTH] (flit_to_south      [i  ][j+1][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_SOUTH] (ack_to_north       [i  ][j+1][k  ]);
																				
			t[i][j][k]->req_tx             [DIRECTION_WEST ] (req_to_west        [i  ][j  ][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_WEST ] (flit_to_west       [i  ][j  ][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_WEST ] (ack_to_east        [i  ][j  ][k  ]);
			
			//To VLink																	
			t[i][j][k]->req_tx             [DIRECTION_UP   ] (req_toV_up         [i  ][j  ][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_UP   ] (flit_toV_up        [i  ][j  ][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_UP   ] (ack_toT_down       [i  ][j  ][k  ]);
			//To VLink														     
			t[i][j][k]->req_tx             [DIRECTION_DOWN ] (req_toV_down       [i  ][j  ][k  ]);
			t[i][j][k]->flit_tx            [DIRECTION_DOWN ] (flit_toV_down      [i  ][j  ][k  ]);
			t[i][j][k]->ack_tx             [DIRECTION_DOWN ] (ack_toT_up         [i  ][j  ][k  ]);
			
			// Map buffer level signals (analogy with req_tx/rx port mapping)
			t[i][j][k]->free_slots         [DIRECTION_NORTH] (free_slots_to_north[i  ][j  ][k  ]);
			t[i][j][k]->free_slots         [DIRECTION_EAST ] (free_slots_to_east [i+1][j  ][k  ]);
			t[i][j][k]->free_slots         [DIRECTION_SOUTH] (free_slots_to_south[i  ][j+1][k  ]);
			t[i][j][k]->free_slots         [DIRECTION_WEST ] (free_slots_to_west [i  ][j  ][k  ]);
			t[i][j][k]->free_slots         [DIRECTION_UP   ] (free_slots_to_up   [i  ][j  ][k  ]);
			t[i][j][k]->free_slots         [DIRECTION_DOWN ] (free_slots_to_down [i  ][j  ][k+1]);
			
			t[i][j][k]->free_slots_neighbor[DIRECTION_NORTH] (free_slots_to_south[i  ][j  ][k  ]);
			t[i][j][k]->free_slots_neighbor[DIRECTION_EAST ] (free_slots_to_west [i+1][j  ][k  ]);
			t[i][j][k]->free_slots_neighbor[DIRECTION_SOUTH] (free_slots_to_north[i  ][j+1][k  ]);
			t[i][j][k]->free_slots_neighbor[DIRECTION_WEST ] (free_slots_to_east [i  ][j  ][k  ]);
			t[i][j][k]->free_slots_neighbor[DIRECTION_UP   ] (free_slots_to_down [i  ][j  ][k  ]);
			t[i][j][k]->free_slots_neighbor[DIRECTION_DOWN ] (free_slots_to_up   [i  ][j  ][k+1]);

			
			// // NoP 
			// t[i][j][k]->NoP_data_out       [DIRECTION_NORTH] (NoP_data_to_north  [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_out       [DIRECTION_EAST ] (NoP_data_to_east   [i+1][j  ][k  ]);
			// t[i][j][k]->NoP_data_out       [DIRECTION_SOUTH] (NoP_data_to_south  [i  ][j+1][k  ]);
			// t[i][j][k]->NoP_data_out       [DIRECTION_WEST ] (NoP_data_to_west   [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_out       [DIRECTION_UP   ] (NoP_data_to_up     [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_out       [DIRECTION_DOWN ] (NoP_data_to_down   [i  ][j  ][k+1]);
																				
			// t[i][j][k]->NoP_data_in        [DIRECTION_NORTH] (NoP_data_to_south  [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_in        [DIRECTION_EAST ] (NoP_data_to_west   [i+1][j  ][k  ]);
			// t[i][j][k]->NoP_data_in        [DIRECTION_SOUTH] (NoP_data_to_north  [i  ][j+1][k  ]);
			// t[i][j][k]->NoP_data_in        [DIRECTION_WEST ] (NoP_data_to_east   [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_in        [DIRECTION_UP   ] (NoP_data_to_down   [i  ][j  ][k  ]);
			// t[i][j][k]->NoP_data_in        [DIRECTION_DOWN ] (NoP_data_to_up     [i  ][j  ][k+1]);
			// Derek @2012.03.07
			// t[i][j][k]->RCA_data_out[DIRECTION_NORTH*2+0](RCA_data_to_north0[i][j][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_NORTH*2+1](RCA_data_to_north1[i][j][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_EAST*2+0](RCA_data_to_east0[i+1][j][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_EAST*2+1](RCA_data_to_east1[i+1][j][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_SOUTH*2+0](RCA_data_to_south0[i][j+1][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_SOUTH*2+1](RCA_data_to_south1[i][j+1][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_WEST*2+0](RCA_data_to_west0[i][j][k]);
			// t[i][j][k]->RCA_data_out[DIRECTION_WEST*2+1](RCA_data_to_west1[i][j][k]);
            
			// t[i][j][k]->RCA_data_in[DIRECTION_NORTH*2+0](RCA_data_to_south1[i][j][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_NORTH*2+1](RCA_data_to_south0[i][j][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_EAST*2+0](RCA_data_to_west1[i+1][j][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_EAST*2+1](RCA_data_to_west0[i+1][j][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_SOUTH*2+0](RCA_data_to_north1[i][j+1][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_SOUTH*2+1](RCA_data_to_north0[i][j+1][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_WEST*2+0](RCA_data_to_east1[i][j][k]);    //***0 1 inverse
			// t[i][j][k]->RCA_data_in[DIRECTION_WEST*2+1](RCA_data_to_east0[i][j][k]);    //***0 1 inverse			
			
			// t[i][j][k]->monitor_out        [DIRECTION_NORTH] (RCA_to_north       [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_out        [DIRECTION_EAST ] (RCA_to_east        [i+1][j  ][k  ]);
			// t[i][j][k]->monitor_out        [DIRECTION_SOUTH] (RCA_to_south       [i  ][j+1][k  ]);
			// t[i][j][k]->monitor_out        [DIRECTION_WEST ] (RCA_to_west        [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_out        [DIRECTION_UP   ] (RCA_to_up          [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_out        [DIRECTION_DOWN ] (RCA_to_down        [i  ][j  ][k+1]);
			// t[i][j][k]->monitor_in         [DIRECTION_NORTH] (RCA_to_south       [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_in         [DIRECTION_EAST ] (RCA_to_west        [i+1][j  ][k  ]);
			// t[i][j][k]->monitor_in         [DIRECTION_SOUTH] (RCA_to_north       [i  ][j+1][k  ]);
			// t[i][j][k]->monitor_in         [DIRECTION_WEST ] (RCA_to_east        [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_in         [DIRECTION_UP   ] (RCA_to_down        [i  ][j  ][k  ]);
			// t[i][j][k]->monitor_in         [DIRECTION_DOWN ] (RCA_to_up          [i  ][j  ][k+1]);
			
			// on/off ports in emergency mode
			t[i][j][k]->on_off             [DIRECTION_NORTH] (on_off_to_north    [i  ][j  ][k  ]);
			t[i][j][k]->on_off             [DIRECTION_EAST ] (on_off_to_east     [i+1][j  ][k  ]);
			t[i][j][k]->on_off             [DIRECTION_SOUTH] (on_off_to_south    [i  ][j+1][k  ]);
			t[i][j][k]->on_off             [DIRECTION_WEST ] (on_off_to_west     [i  ][j  ][k  ]);
			t[i][j][k]->on_off             [DIRECTION_UP ]    (on_off_to_up      [i  ][j  ][k  ]);
			t[i][j][k]->on_off             [DIRECTION_DOWN ] (on_off_to_down    [i  ][j  ][k+1]);
			
			t[i][j][k]->on_off_neighbor    [DIRECTION_NORTH] (on_off_to_south    [i  ][j  ][k  ]);
			t[i][j][k]->on_off_neighbor    [DIRECTION_EAST ] (on_off_to_west     [i+1][j  ][k  ]);
			t[i][j][k]->on_off_neighbor    [DIRECTION_SOUTH] (on_off_to_north    [i  ][j+1][k  ]);
			t[i][j][k]->on_off_neighbor    [DIRECTION_WEST ] (on_off_to_east     [i  ][j  ][k  ]);
			t[i][j][k]->on_off_neighbor    [DIRECTION_UP]    (on_off_to_down     [i  ][j  ][k  ]);
			t[i][j][k]->on_off_neighbor    [DIRECTION_DOWN ] (on_off_to_up       [i  ][j  ][k+1]);
			
			
			// Thermal budget
			// t[i][j][k]->TB             [DIRECTION_NORTH] (TB_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->TB             [DIRECTION_EAST ] (TB_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->TB             [DIRECTION_SOUTH] (TB_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->TB             [DIRECTION_WEST ] (TB_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->TB             [DIRECTION_UP ]    (TB_to_up      [i  ][j  ][k  ]);
			// t[i][j][k]->TB             [DIRECTION_DOWN ] (TB_to_down    [i  ][j  ][k+1]);
			
			// t[i][j][k]->TB_neighbor    [DIRECTION_NORTH] (TB_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->TB_neighbor    [DIRECTION_EAST ] (TB_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->TB_neighbor    [DIRECTION_SOUTH] (TB_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->TB_neighbor    [DIRECTION_WEST ] (TB_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->TB_neighbor    [DIRECTION_UP]    (TB_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->TB_neighbor    [DIRECTION_DOWN ] (TB_to_up       [i  ][j  ][k+1]);
		
			// // predict temp.
			// t[i][j][k]->PDT             [DIRECTION_NORTH] (PDT_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->PDT             [DIRECTION_EAST ] (PDT_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->PDT             [DIRECTION_SOUTH] (PDT_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->PDT             [DIRECTION_WEST ] (PDT_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->PDT             [DIRECTION_UP ]   (PDT_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->PDT             [DIRECTION_DOWN ] (PDT_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->PDT_neighbor    [DIRECTION_NORTH] (PDT_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->PDT_neighbor    [DIRECTION_EAST ] (PDT_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->PDT_neighbor    [DIRECTION_SOUTH] (PDT_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->PDT_neighbor    [DIRECTION_WEST ] (PDT_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->PDT_neighbor    [DIRECTION_UP]    (PDT_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->PDT_neighbor    [DIRECTION_DOWN ] (PDT_to_up       [i  ][j  ][k+1]);
	
			// //buffer information
			// t[i][j][k]->buf[0]             [DIRECTION_NORTH] (buf0_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[0]             [DIRECTION_EAST ] (buf0_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[0]             [DIRECTION_SOUTH] (buf0_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[0]             [DIRECTION_WEST ] (buf0_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[0]             [DIRECTION_UP ]   (buf0_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[0]             [DIRECTION_DOWN ] (buf0_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_NORTH] (buf0_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_EAST ] (buf0_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_SOUTH] (buf0_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_WEST ] (buf0_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_UP]    (buf0_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[0]    [DIRECTION_DOWN ] (buf0_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[1]             [DIRECTION_NORTH] (buf1_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[1]             [DIRECTION_EAST ] (buf1_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[1]             [DIRECTION_SOUTH] (buf1_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[1]             [DIRECTION_WEST ] (buf1_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[1]             [DIRECTION_UP ]   (buf1_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[1]             [DIRECTION_DOWN ] (buf1_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_NORTH] (buf1_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_EAST ] (buf1_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_SOUTH] (buf1_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_WEST ] (buf1_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_UP]    (buf1_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[1]    [DIRECTION_DOWN ] (buf1_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[2]             [DIRECTION_NORTH] (buf2_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[2]             [DIRECTION_EAST ] (buf2_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[2]             [DIRECTION_SOUTH] (buf2_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[2]             [DIRECTION_WEST ] (buf2_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[2]             [DIRECTION_UP ]   (buf2_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[2]             [DIRECTION_DOWN ] (buf2_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_NORTH] (buf2_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_EAST ] (buf2_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_SOUTH] (buf2_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_WEST ] (buf2_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_UP]    (buf2_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[2]    [DIRECTION_DOWN ] (buf2_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[3]             [DIRECTION_NORTH] (buf3_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[3]             [DIRECTION_EAST ] (buf3_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[3]             [DIRECTION_SOUTH] (buf3_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[3]             [DIRECTION_WEST ] (buf3_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[3]             [DIRECTION_UP ]   (buf3_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[3]             [DIRECTION_DOWN ] (buf3_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_NORTH] (buf3_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_EAST ] (buf3_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_SOUTH] (buf3_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_WEST ] (buf3_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_UP]    (buf3_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[3]    [DIRECTION_DOWN ] (buf3_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[4]             [DIRECTION_NORTH] (buf4_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[4]             [DIRECTION_EAST ] (buf4_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[4]             [DIRECTION_SOUTH] (buf4_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[4]             [DIRECTION_WEST ] (buf4_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[4]             [DIRECTION_UP ]   (buf4_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[4]             [DIRECTION_DOWN ] (buf4_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_NORTH] (buf4_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_EAST ] (buf4_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_SOUTH] (buf4_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_WEST ] (buf4_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_UP]    (buf4_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[4]    [DIRECTION_DOWN ] (buf4_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[5]             [DIRECTION_NORTH] (buf5_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[5]             [DIRECTION_EAST ] (buf5_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[5]             [DIRECTION_SOUTH] (buf5_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[5]             [DIRECTION_WEST ] (buf5_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[5]             [DIRECTION_UP ]   (buf5_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[5]             [DIRECTION_DOWN ] (buf5_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_NORTH] (buf5_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_EAST ] (buf5_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_SOUTH] (buf5_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_WEST ] (buf5_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_UP]    (buf5_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[5]    [DIRECTION_DOWN ] (buf5_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[6]             [DIRECTION_NORTH] (buf6_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[6]             [DIRECTION_EAST ] (buf6_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[6]             [DIRECTION_SOUTH] (buf6_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[6]             [DIRECTION_WEST ] (buf6_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[6]             [DIRECTION_UP ]   (buf6_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[6]             [DIRECTION_DOWN ] (buf6_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_NORTH] (buf6_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_EAST ] (buf6_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_SOUTH] (buf6_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_WEST ] (buf6_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_UP]    (buf6_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[6]    [DIRECTION_DOWN ] (buf6_to_up       [i  ][j  ][k+1]);

			// //buffer information
			// t[i][j][k]->buf[7]             [DIRECTION_NORTH] (buf7_to_north    [i  ][j  ][k  ]);
			// t[i][j][k]->buf[7]             [DIRECTION_EAST ] (buf7_to_east     [i+1][j  ][k  ]);
			// t[i][j][k]->buf[7]             [DIRECTION_SOUTH] (buf7_to_south    [i  ][j+1][k  ]);
			// t[i][j][k]->buf[7]             [DIRECTION_WEST ] (buf7_to_west     [i  ][j  ][k  ]);
			// t[i][j][k]->buf[7]             [DIRECTION_UP ]   (buf7_to_up       [i  ][j  ][k  ]);
			// t[i][j][k]->buf[7]             [DIRECTION_DOWN ] (buf7_to_down     [i  ][j  ][k+1]);

			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_NORTH] (buf7_to_south    [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_EAST ] (buf7_to_west     [i+1][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_SOUTH] (buf7_to_north    [i  ][j+1][k  ]);
			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_WEST ] (buf7_to_east     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_UP]    (buf7_to_down     [i  ][j  ][k  ]);
			// t[i][j][k]->buf_neighbor[7]    [DIRECTION_DOWN ] (buf7_to_up       [i  ][j  ][k+1]);

			//NoximVlink 
			if( k < NoximGlobalParams::mesh_dim_z - 1){
				v[i][j]   ->ack_rx_to_UP   [k]( ack_toV_down  [i][j][k  ] );
				v[i][j]   ->req_rx_to_UP   [k]( req_toT_up    [i][j][k  ] );//output
				v[i][j]   ->flit_rx_to_UP  [k]( flit_toT_up   [i][j][k  ] );//output
				v[i][j]   ->ack_tx_to_UP   [k]( ack_toT_up    [i][j][k  ] );//output
				v[i][j]   ->req_tx_to_UP   [k]( req_toV_down  [i][j][k  ] );
				v[i][j]   ->flit_tx_to_UP  [k]( flit_toV_down [i][j][k  ] );
				
				v[i][j]   ->ack_rx_to_DOWN [k]( ack_toT_down  [i][j][k+1] );//output
				v[i][j]   ->req_rx_to_DOWN [k]( req_toV_up    [i][j][k+1] );
				v[i][j]   ->flit_rx_to_DOWN[k]( flit_toV_up   [i][j][k+1] );
				v[i][j]   ->ack_tx_to_DOWN [k]( ack_toV_up    [i][j][k+1] );
				v[i][j]   ->req_tx_to_DOWN [k]( req_toT_down  [i][j][k+1] );//output
				v[i][j]   ->flit_tx_to_DOWN[k]( flit_toT_down [i][j][k+1] );//output
			}
			//temp_budget[i][j][k]=10;
		}
	}
	
    // dummy NoximNoP_data structure
	NoximNoP_data tmp_NoP;
    tmp_NoP.sender_id = NOT_VALID;

    for ( i = 0; i < DIRECTIONS; i++) {
	tmp_NoP.channel_status_neighbor[i].free_slots = NOT_VALID;
	tmp_NoP.channel_status_neighbor[i].available  = false;
    }

    // Clear signals for borderline nodes
	for( i=0; i<=NoximGlobalParams::mesh_dim_x; i++){
	for( k=0; k<=NoximGlobalParams::mesh_dim_z; k++){
		j = NoximGlobalParams::mesh_dim_y;
		req_to_south       [i][0][k] = 0;
		ack_to_north       [i][0][k] = 0;
		req_to_north       [i][j][k] = 0;
		ack_to_south       [i][j][k] = 0;
		
		free_slots_to_south[i][0][k].write(NOT_VALID);
		free_slots_to_north[i][j][k].write(NOT_VALID);
    
		// RCA_to_south       [i][0][k].write(0);
		// RCA_to_north       [i][j][k].write(0);
    
		// // RCA Derek
		// RCA_data_to_south0[i][0][k].write(0);
		// RCA_data_to_south1[i][0][k].write(0);
		// RCA_data_to_north0[i][NoximGlobalParams::mesh_dim_y][k].write(0);
		// RCA_data_to_north1[i][NoximGlobalParams::mesh_dim_y][k].write(0);					
					
	
		on_off_to_south    [i][0][k].write(NOT_VALID);
		on_off_to_north    [i][j][k].write(NOT_VALID);
		
		// TB_to_south    [i][0][k].write(NOT_VALID);
		// TB_to_north    [i][j][k].write(NOT_VALID);

		// buf0_to_south    [i][0][k].write(NOT_VALID); 
        //         buf0_to_north    [i][j][k].write(NOT_VALID);

		// buf1_to_south    [i][0][k].write(NOT_VALID);
        //         buf1_to_north    [i][j][k].write(NOT_VALID);

		// buf2_to_south    [i][0][k].write(NOT_VALID);
        //         buf2_to_north    [i][j][k].write(NOT_VALID);

		// buf3_to_south    [i][0][k].write(NOT_VALID);
        //         buf3_to_north    [i][j][k].write(NOT_VALID);

		// buf4_to_south    [i][0][k].write(NOT_VALID);
        //         buf4_to_north    [i][j][k].write(NOT_VALID);

		// buf5_to_south    [i][0][k].write(NOT_VALID);
        //         buf5_to_north    [i][j][k].write(NOT_VALID);

		// buf6_to_south    [i][0][k].write(NOT_VALID);
        //         buf6_to_north    [i][j][k].write(NOT_VALID);

		// buf7_to_south    [i][0][k].write(NOT_VALID);
        //         buf7_to_north    [i][j][k].write(NOT_VALID);

		// PDT_to_south    [i][0][k].write(100);
        //         PDT_to_north    [i][j][k].write(100);
    
		// NoP_data_to_south  [i][0][k].write(tmp_NoP);
		// NoP_data_to_north  [i][j][k].write(tmp_NoP);
		}
    }
	for( j=0; j<=NoximGlobalParams::mesh_dim_y; j++)
	for( k=0; k<=NoximGlobalParams::mesh_dim_z; k++){
			i = NoximGlobalParams::mesh_dim_x;
			req_to_east       [0][j][k] = 0;
			ack_to_west       [0][j][k] = 0;
			req_to_west       [i][j][k] = 0;
			ack_to_east       [i][j][k] = 0;

			free_slots_to_east[0][j][k].write(NOT_VALID);
			free_slots_to_west[i][j][k].write(NOT_VALID);
	
			// RCA 
			// RCA_data_to_east0[0][j][k].write(0);
			// RCA_data_to_east1[0][j][k].write(0);
			// RCA_data_to_west0[NoximGlobalParams::mesh_dim_x][j][k].write(0);
			// RCA_data_to_west1[NoximGlobalParams::mesh_dim_x][j][k].write(0);					
					
			// RCA_to_east       [0][j][k].write(0);
			// RCA_to_west       [i][j][k].write(0);

			on_off_to_east    [0][j][k].write(NOT_VALID);
			on_off_to_west    [i][j][k].write(NOT_VALID);
			
			// TB_to_east    [0][j][k].write(NOT_VALID);
			// TB_to_west    [i][j][k].write(NOT_VALID);

			// buf0_to_east    [0][j][k].write(NOT_VALID);
            //             buf0_to_west    [i][j][k].write(NOT_VALID);

			// buf1_to_east    [0][j][k].write(NOT_VALID);
            //             buf1_to_west    [i][j][k].write(NOT_VALID);

			// buf2_to_east    [0][j][k].write(NOT_VALID);
            //             buf2_to_west    [i][j][k].write(NOT_VALID);

			// buf3_to_east    [0][j][k].write(NOT_VALID);
            //             buf3_to_west    [i][j][k].write(NOT_VALID);

			// buf4_to_east    [0][j][k].write(NOT_VALID);
            //             buf4_to_west    [i][j][k].write(NOT_VALID);

			// buf5_to_east    [0][j][k].write(NOT_VALID);
            //             buf5_to_west    [i][j][k].write(NOT_VALID);

			// buf6_to_east    [0][j][k].write(NOT_VALID);
            //             buf6_to_west    [i][j][k].write(NOT_VALID);

			// buf7_to_east    [0][j][k].write(NOT_VALID);
            //             buf7_to_west    [i][j][k].write(NOT_VALID);

			// PDT_to_east    [0][j][k].write(100);
            //             PDT_to_west    [i][j][k].write(100);

			// NoP_data_to_east  [0][j][k].write(tmp_NoP);
			// NoP_data_to_west  [i][j][k].write(tmp_NoP);
	}
	for( i=0; i<=NoximGlobalParams::mesh_dim_x; i++){
	for( j=0; j<=NoximGlobalParams::mesh_dim_y; j++){
		k = NoximGlobalParams::mesh_dim_z;
		req_toT_down       [i][j][0] = 0;
		ack_toT_up         [i][j][k-1] = 0;
		req_toT_up         [i][j][k-1] = 0;
		ack_toT_down       [i][j][0] = 0;
    
		free_slots_to_down[i][j][0].write(NOT_VALID);
		free_slots_to_up  [i][j][k].write(NOT_VALID);
    
		// RCA_to_down       [i][j][0].write(0);
		// RCA_to_up         [i][j][k].write(0);
    
		// NoP_data_to_down  [i][j][0].write(tmp_NoP);
		// NoP_data_to_up    [i][j][k].write(tmp_NoP);
		
		on_off_to_down    [i][j][0].write(NOT_VALID);
		on_off_to_up    [i][j][k].write(NOT_VALID);

		// TB_to_down    [i][j][0].write(NOT_VALID);
		// TB_to_up    [i][j][k].write(NOT_VALID);

		// buf0_to_down    [i][j][0].write(NOT_VALID);
        //         buf0_to_up    [i][j][k].write(NOT_VALID);

		// buf1_to_down    [i][j][0].write(NOT_VALID);
        //         buf1_to_up    [i][j][k].write(NOT_VALID);

		// buf2_to_down    [i][j][0].write(NOT_VALID);
        //         buf2_to_up    [i][j][k].write(NOT_VALID);

		// buf3_to_down    [i][j][0].write(NOT_VALID);
        //         buf3_to_up    [i][j][k].write(NOT_VALID);

		// buf4_to_down    [i][j][0].write(NOT_VALID);
        //         buf4_to_up    [i][j][k].write(NOT_VALID);

		// buf5_to_down    [i][j][0].write(NOT_VALID);
        //         buf5_to_up    [i][j][k].write(NOT_VALID);

		// buf6_to_down    [i][j][0].write(NOT_VALID);
        //         buf6_to_up    [i][j][k].write(NOT_VALID);

		// buf7_to_down    [i][j][0].write(NOT_VALID);
        //         buf7_to_up    [i][j][k].write(NOT_VALID);
	
		// PDT_to_down    [i][j][0].write(100);
        //         PDT_to_up    [i][j][k].write(100);
		}
    }

    // invalidate reservation table entries for non-exhistent channels
 	for( i=0; i<NoximGlobalParams::mesh_dim_x; i++)
	for( k=0; k<NoximGlobalParams::mesh_dim_z; k++){
		j = NoximGlobalParams::mesh_dim_y;
		t[i][0  ][k]->r->reservation_table.invalidate(DIRECTION_NORTH);
		t[i][j-1][k]->r->reservation_table.invalidate(DIRECTION_SOUTH);
	}
	for( j=0; j<NoximGlobalParams::mesh_dim_y; j++)
	for( k=0; k<NoximGlobalParams::mesh_dim_z; k++){
		i = NoximGlobalParams::mesh_dim_x;
		t[0  ][j][k]->r->reservation_table.invalidate(DIRECTION_WEST);
		t[i-1][j][k]->r->reservation_table.invalidate(DIRECTION_EAST);
	}   
	
	// for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
	// for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++)
	// for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++){
	// 	t[x][y][z]->vertical_free_slot_out(vertical_free_slot[x][y][z]);
	// 	for( int i=0; i < NoximGlobalParams::mesh_dim_z; i++ )
	// 		t[x][y][z]->vertical_free_slot_in[i]( vertical_free_slot[x][y][i] );
	// }
	
	// Initial emergency mode 
	if(NoximGlobalParams::throt_type == THROT_TEST){
		cout<<"building throt sest!"<<endl;
		_throt_case_setting(NoximGlobalParams::dynamic_throt_case);
	}
	else{
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			throttling2[i][j][k] = 0;
			_setNormal(i,j,k);		
		}
	}

	int non_beltway_layer,non_throt_layer;
	int col_max,col_min,row_max,row_min;
	findNonXLayer(non_throt_layer,non_beltway_layer);
	calROC(col_max,col_min,row_max,row_min,non_beltway_layer);
	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++ )
	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++ ){
		t[x][y][z]->pe->RTM_set_var(non_throt_layer, non_beltway_layer, NoximGlobalParams::ROC_UP, NoximGlobalParams::ROC_DOWN, NoximGlobalParams::ROC_UP, NoximGlobalParams::ROC_DOWN);
	}
	//cout<<col_max<<" "<<col_min<<" "<<row_max<<" "<<row_min;
}

void NoximNoC::entry(){  //Foster big modified - 09/11/12
	//reset power
	if (reset.read()) {
		//in reset phase, reset power value 
		for(int k=0; k < NoximGlobalParams::mesh_dim_z; k++)
		for(int j=0; j < NoximGlobalParams::mesh_dim_y; j++)	
		for(int i=0; i < NoximGlobalParams::mesh_dim_x; i++){		
			t[i][j][k]->r->stats.power.resetPwr();
			t[i][j][k]->r->stats.power.resetTransientPwr();
			t[i][j][k]->r->stats.temperature = INIT_TEMP - 273.15;
			t[i][j][k]->r->stats.pre_temperature1 = INIT_TEMP - 273.15;
			//MTTT[i][j][k] = 10;
			traffic[i][j][k] = 0;
			traffic2[i][j][k] = 0;
		}
		//the variable for interlace
		MaxTempall=0;
		interlace_type=false;
		rema=0;
		arr = read_3d_array_from_file("output.txt");
		// 打印读取到的三维数组
		for (size_t i = 0; i < arr.size(); ++i) {
			std::cout << "--- Slice " << i << " ---" << std::endl;
			for (const auto& row : arr[i]) {
				for (int num : row) {
					std::cout << num << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout <<arr.size() <<std::endl;
		std::cout << arr[0].size()<<std::endl;
		std::cout << arr[0][0].size()<<std::endl;
		if(arr.size()!=NoximGlobalParams::mesh_dim_z||arr[0].size()!=NoximGlobalParams::mesh_dim_y||arr[0][0].size()!=NoximGlobalParams::mesh_dim_y){
			std::cout<<"Error:Dynamic array dimensions do not match"<<std::endl;
		}
		for(size_t z=0;z<NoximGlobalParams::mesh_dim_z;z++){
			for(size_t y=0;y<NoximGlobalParams::mesh_dim_y;y++){
				for(size_t x=0;x<NoximGlobalParams::mesh_dim_x;x++){
					best_grid[z][y][x]= arr[z][y][x];
				}	
			}
		}
		std::cerr<<"success"<<std::endl;
		//
		_emergency = false;
		_clean     = true;
	/*	if(!mkdir("results/Traffic",0777)) cout<<"Making new directory results/Hist"<<endl;
		string filename;
		filename = "results/Traffic/Traffic_analysis";
		filename = MarkFileName( filename );
	*/
	}
	else{
		int CurrentCycle    = getCurrentCycleNum();
		int CurrentCycleMod = (getCurrentCycleNum() % (int) (TEMP_REPORT_PERIOD));
		if(  CurrentCycleMod == ((int) (TEMP_REPORT_PERIOD) - NoximGlobalParams::clean_stage_time)){
			cout<<"*****setCleanStage****"<<endl;
            //TransientLog();
			setCleanStage();
			//num_pkt = 7168800;
		}
		//cout<<"CurrentCycle:"<<CurrentCycle<<"\r"<<flush;
		if( CurrentCycleMod == 0 ){
			EndCleanStage();
			
			//accumulate steady power after warm-up time
			if( CurrentCycle > (int)( NoximGlobalParams::stats_warm_up_time ) )
				steadyPwr2PtraceFile();
			//Calculate Temperature
			if( NoximGlobalParams::cal_temp ){
				transPwr2PtraceFile();
				HS_interface->Temperature_calc(instPowerTrace, TemperatureTrace);
				//the function for prediction
				if(NoximGlobalParams::pre){
					LSTMpre();
					cout<<"Predictive DTM"<<endl;}
				else
					setTemperature();
				setTemperature();
			}
			//check temperature, whether set emergency mode or not
			EmergencyDecision();
			//@ct: throttling test
			//if(CurrentCycle == 1000000)
			//	_setThrot(3,3,0);
			//else if(CurrentCycle == 1100000)
			//	_setNormal(3,3,0);	
			cout<<"*****EndCleanStage*****"<<endl;
			PRETempLog();		
			TransientLog();
			//the function for interlace
			//interlace_type 没5_0000个周期反转一次，下一次进行温度控制的就是下一类结点了
			interlace_type=!interlace_type;
			//remainder 每50000个周期增加一个，等于设定值时清零
			if(rema<NoximGlobalParams::classification-1){
				rema++;
			}
			else{
				rema=0;
			}
		/*
		// debug---------------------------------
		// Derek 2012.10.16 
		cout << "--------------Temperature Difference/ Location Factor in Entry Function--------------"<<endl;
		int m, n, o;
		for( o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){			
		for( n = NoximGlobalParams::mesh_dim_y - 1 ; n > -2  ; n-- ){
		for( m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
			if( n != -1 ){
					 cout << t[m][n][o]->r->stats.pre_temperature1 <<"\t";	
					}
					else
						cout << "0\t";
				}
				cout << "0\n";
			}
			cout<<"]"<<endl;
		}
		//-----------------------------------------
			
		cout << "--------------Buffer log--------------"<<endl;
		for(int i=0 ; i<6 ; i++){
			cout <<"Direction"<<i<<endl;
                for( o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){
                for( n = NoximGlobalParams::mesh_dim_y - 1 ; n > -2  ; n-- ){
                for( m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
                        if( n != -1 ){
                                         cout << t[m][n][o]->r->buffer[i].GetMaxBufferSize() <<"\t";
                                        }
                                        else
                                                cout << "0\t";
                                }
                                cout << "0\n";
                        }
                        cout<<"]"<<endl;
                }}
                //-----------------------------------------
		*/
		}

		//trun off the throttling mark based on throt level

		if( CurrentCycle == NoximGlobalParams::simulation_time && NoximGlobalParams::cal_temp){ //Calculate steady state temp.
			cout<<"Calculate SteadyTemp at "<<getCurrentCycleNum()<<endl;
			HS_interface->steadyTmp(t);
		} 

		//@ct new FGR: trun off the throttling mask at certain timepoint based on the throtlting level
		int level_period = (int) TEMP_REPORT_PERIOD/4;  //4 is the size of your throttling level 
		if(CurrentCycle % level_period == 0){
			for(int k=0; k < NoximGlobalParams::mesh_dim_z; k++)
			for(int j=0; j < NoximGlobalParams::mesh_dim_y; j++)	
			for(int i=0; i < NoximGlobalParams::mesh_dim_x; i++){		
				int time_division = CurrentCycleMod/level_period; 
				if( throttling2[i][j][k]>0 && (time_division == throttling2[i][j][k]) )
					throttling[i][j][k] = false;
			}
		}

	
		/*
		int CurrentCycle    = getCurrentCycleNum();
		int CurrentCycleMod = (getCurrentCycleNum() % (int) (TEMP_REPORT_PERIOD));
		cout<<"CurrentCycle:"<<CurrentCycle<<"\r"<<flush;
		//if(  CurrentCycleMod == ((int) (TEMP_REPORT_PERIOD) - NoximGlobalParams::clean_stage_time)){
		//	setCleanStage();
		//}
		if( CurrentCycleMod == 0 ){
			//EndCleanStage();
			//accumulate steady power after warm-up time
			if( CurrentCycle > (int)( NoximGlobalParams::stats_warm_up_time ) )
				steadyPwr2PtraceFile();
			//Calculate Temperature
			if( NoximGlobalParams::cal_temp ){
				transPwr2PtraceFile();
				HS_interface->Temperature_calc(instPowerTrace, TemperatureTrace);
				setTemperature();
				cout<<getCurrentCycleNum()<<":Calc. Temp."<<endl;
			}
			//check temperature, whether set emergency mode or not
			//_emergency = EmergencyDecision();
			//if ( _emergency ){ 
				setCleanStage();
				_clean = false;
			//}
			//else {
			//	_clean = true;
			//	TransientLog();
			//}
			//TransientLog();
		}
		if( !_clean &&  ( CurrentCycleMod %  NoximGlobalParams::clean_stage_time == 0) ){
			if ( _CleanDone() ){
				EndCleanStage();
				_emergency = EmergencyDecision();
				TransientLog();
				_clean = true;
			}
			else{
				cout<<getCurrentCycleNum()<<":Clean stage fail."<<endl;
				_clean = false;
			}
		}
		if( CurrentCycle == NoximGlobalParams::simulation_time && NoximGlobalParams::cal_temp){ //Calculate steady state temp.
			cout<<"Calculate SteadyTemp at "<<getCurrentCycleNum()<<endl;
			HS_interface->steadyTmp(t);
		} 
	*/
		if(CurrentCycleMod < ((int) (TEMP_REPORT_PERIOD) - NoximGlobalParams::clean_stage_time)){
	    
			if(getCurrentCycleNum() % 100000 == 0){
				int increment;

				traffic_analysis<<" -------------- Major temper sampling :"<<getCurrentCycleNum()<<" -------------- \n";
				traffic_analysis<<"total traffic\n";
				for( int o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){
					traffic_analysis<<"XY"<<o<<" = ["<<"\n";
					for( int n = NoximGlobalParams::mesh_dim_y -1 ; n>-1 ; n-- ){
						for( int m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
							increment = t[m][n][o]->r->getRoutedFlits() - traffic2[m][n][o];
							traffic_analysis<<increment<<"\t";
						}
							
						traffic_analysis<<"\n";
					}
					traffic_analysis<<"]\n"<<"\n";
				}

				traffic_analysis<<"color_range = [0 300000]"<<endl;
				traffic_analysis<<"figure(1)"<<endl;

				int temp = 1;
				for( int k = 0 ; k < NoximGlobalParams::mesh_dim_z ; k++){
						traffic_analysis<<"subplot("<<NoximGlobalParams::mesh_dim_z<<",1,"<<temp<<"), pcolor(XY"<<k<<"), axis off, caxis( color_range ), colormap(jet)"<<endl;
						temp += 1;
				}
				traffic_analysis<<"set(gcf, 'PaperPosition', [1 1 7 30]);"<<endl;
				traffic_analysis<<"print(gcf,'-djpeg','-r0','"<<MarkFileName( string("") )<<".jpg')"<<endl;

				for( int o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){
					for( int n = 0 ; n < NoximGlobalParams::mesh_dim_y ; n++ ){
						for( int m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
							traffic2[m][n][o] = t[m][n][o]->r->getRoutedFlits();
							traffic[m][n][o]  = traffic2[m][n][o];
						}	
					}
				}

			}
			else if(getCurrentCycleNum() % 10000 == 0){
				
				throt_analysis<<"Cycletime: "<<getCurrentCycleNum()<<"\n";
				for(int o=0; o < NoximGlobalParams::mesh_dim_z; o++){
					throt_analysis<<"XY"<<o<<"=[\n";
					for(int n=NoximGlobalParams::mesh_dim_y-1; n > -1; n--){
						for(int m=0; m < NoximGlobalParams::mesh_dim_x; m++){
							throt_analysis<< throttling[m][n][o] << "\t";
						}
						throt_analysis<<"\n";
					}
					throt_analysis<<"]\n"<<"\n";
				}
				throt_analysis.flush();


				traffic_analysis<<"minor Cycletime: "<<getCurrentCycleNum()<<"\n";

				int increment;

				for( int o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){
					traffic_analysis<<"XY"<<o<<"=[\n";
					for( int n = NoximGlobalParams::mesh_dim_y -1 ; n>-1 ; n-- ){
						for( int m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
							increment = t[m][n][o]->r->getRoutedFlits() - traffic[m][n][o];
							traffic_analysis<<increment<<"\t";
						}
					traffic_analysis<<"\n";
					}
					traffic_analysis<<"]\n"<<"\n";
				}
				
				for( int o = 0 ; o < NoximGlobalParams::mesh_dim_z ; o++ ){
				for( int n = 0 ; n < NoximGlobalParams::mesh_dim_y ; n++ ){
				for( int m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ )
					traffic[m][n][o] = t[m][n][o]->r->getRoutedFlits();
				}
				}

			}
		}
	
	}      
}

NoximTile *NoximNoC::searchNode(const int id) const{
	int i,j,k;
	NoximCoord node = id2Coord(id);
	return t[node.x][node.y][node.z];
}

//----------Modified by CMH
void NoximNoC::transPwr2PtraceFile()
{
    int idx = 0;	
	int m, n, o;
	/*================================Begin of collecting POWER TRACE ======================================*/
	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++)
	for(n=0; n < NoximGlobalParams::mesh_dim_y; n++)
	for(m=0; m < NoximGlobalParams::mesh_dim_x; m++){
		idx = xyz2Id( m, n, o);
		
		double a = t[m][n][o]->r->stats.power.getTransientRouterPower();
		//router : offset = 0
		//instPowerTrace[3*idx] = t[m][n][o]->r->stats.power.getTransientRouterPower()/(TEMP_REPORT_PERIOD *1e-9);
		//overallPowerTrace[3*idx] += instPowerTrace[3*idx];
		instPowerTrace[3*idx] = t[m][n][o]->r->stats.power.getTransientRouterPower();
		results_log_pwr << instPowerTrace[3*idx]<<"\t";	
				
        //uP_mem : offset = 1
		//instPowerTrace[3*idx+1] = t[m][n][o]->r->stats.power.getTransientMEM()/(TEMP_REPORT_PERIOD *1e-9);
		//overallPowerTrace[3*idx+1] += instPowerTrace[3*idx+1];
		instPowerTrace[3*idx+1] = t[m][n][o]->r->stats.power.getTransientMEMPower();
		results_log_pwr << instPowerTrace[3*idx+1]<<"\t";	

		//uP_mac : offset = 2
		//instPowerTrace[3*idx+2] = t[m][n][o]->r->stats.power.getTransientFPMACPower()/(TEMP_REPORT_PERIOD *1e-9);
		//overallPowerTrace[3*idx+2] += instPowerTrace[3*idx+2];
		instPowerTrace[3*idx+2] = t[m][n][o]->r->stats.power.getTransientFPMACPower();
		results_log_pwr << instPowerTrace[3*idx+2]<<"\t";	

    	t[m][n][o]->r->stats.power.resetTransientPwr();
	}
	/*================================End of COLLECTING Power TRACE=================================================*/
	results_log_pwr<<"\n";
}

void NoximNoC::steadyPwr2PtraceFile()
{
    int idx = 0;	
	int m, n, o;
	/*================================Begin of collecting POWER TRACE ======================================*/
	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++)
	for(n=0; n < NoximGlobalParams::mesh_dim_y; n++)
	for(m=0; m < NoximGlobalParams::mesh_dim_x; m++){
        idx = xyz2Id( m, n, o);
		//router : offset = 0
		overallPowerTrace[3*idx  ] += t[m][n][o]->r->stats.power.getSteadyStateRouterPower();					
        //uP_mem : offset = 1
		overallPowerTrace[3*idx+1] += t[m][n][o]->r->stats.power.getSteadyStateMEMPower   ();
		//uP_mac : offset = 2
		overallPowerTrace[3*idx+2] += t[m][n][o]->r->stats.power.getSteadyStateFPMACPower ();
	}
	/*================================End of COLLECTING Power TRACE=================================================*/
}

void NoximNoC::setTemperature(){
	int m, n, o;
    int idx = 0;
	// temperature prediction-----
	double current_temp; 
	double current_delta_temp; 
	double pre_delta_temp; 
	double pre_current_temp;
	double adjustment; 
	//double consumption_rate[20][20][4];	
	
	
	
	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++)
	for(n=0; n < NoximGlobalParams::mesh_dim_y; n++) 
	for(m=0; m < NoximGlobalParams::mesh_dim_x; m++) {
		idx = xyz2Id( m, n, o);
		//set tile temperature
		t[m][n][o]->r->stats.last_temperature = t[m][n][o]->r->stats.temperature;
		t[m][n][o]->r->stats.temperature      = TemperatureTrace[3*idx];     

		//t[m][n][o]->r->TBDB(consumption_rate[m][n][o]);

        //thermal budget
		// temp_budget[m][n][o]            	  = TEMP_THRESHOLD - t[m][n][o]->r->stats.temperature; // Derek 2012.10.16 	
		// if (temp_budget[m][n][o]<0)
		// 	temp_budget[m][n][o] = 0;	
		//thermal prediction

		//if(t[m][n][o]->r->stats.temperature > 85)
		current_delta_temp = t[m][n][o]->r->stats.temperature - t[m][n][o]->r->stats.last_temperature;
		if(throttling2[m][n][o] ==0 && current_delta_temp>0)
			deltaT[m][n][o] = current_delta_temp;  //keep recording the temp increase amptitude for providing Criteria in FGR

		if(getCurrentCycleNum()/TEMP_REPORT_PERIOD > 5)
		{
			current_temp = t[m][n][o]->r->stats.temperature;
			

			if(current_delta_temp < 0){
				pre_delta_temp = t[m][n][o]->r->stats.last_pre_temperature1 - t[m][n][o]->r->stats.last_temperature;
				pre_current_temp = t[m][n][o]->r->stats.last_pre_temperature1;
				adjustment = t[m][n][o]->r->stats.last_pre_temperature1 - current_temp;
				
				t[m][n][o]->r->stats.pre_temperature1 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) - adjustment;
				//if(TEMP_THRESHOLD - t[m][n][o]->r->stats.pre_temperature1>0)
				//consumption_rate[m][n][o] = t[m][n][o]->r->stats.pre_temperature1 - t[m][n][o]->r->stats.temperature; // Jason
				//else
				//	temp_budget[m][n][o]=0;
				t[m][n][o]->r->stats.pre_temperature2 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) + pre_delta_temp* exp(-1.98*0.02) - adjustment;
				t[m][n][o]->r->stats.pre_temperature3 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) + pre_delta_temp* exp(-1.98*0.02) +
				                                         pre_delta_temp* exp(-1.98*0.03) - adjustment;
				t[m][n][o]->r->stats.pre_temperature4 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) + pre_delta_temp* exp(-1.98*0.02) +
				                                         pre_delta_temp* exp(-1.98*0.03) + pre_delta_temp* exp(-1.98*0.04) - adjustment;
				t[m][n][o]->r->stats.pre_temperature5 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) + pre_delta_temp* exp(-1.98*0.02) +
				                                         pre_delta_temp* exp(-1.98*0.03) + pre_delta_temp* exp(-1.98*0.04) + 
														 pre_delta_temp* exp(-1.98*0.05) - adjustment;
				t[m][n][o]->r->stats.pre_temperature6 =  pre_current_temp + pre_delta_temp* exp(-1.98*0.01) + pre_delta_temp* exp(-1.98*0.02) +
				                                         pre_delta_temp* exp(-1.98*0.03) + pre_delta_temp* exp(-1.98*0.04) + 
														 pre_delta_temp* exp(-1.98*0.05) + pre_delta_temp* exp(-1.98*0.06) - adjustment;														 
			
			}
			//else t[m][n][o]->r->stats.pre_temperature =  current_temp + current_delta_temp* exp(-2.95*0.01) + current_delta_temp* exp(-2.95*0.02) + current_delta_temp* exp(-2.95*0.03) + current_delta_temp*exp(-2.95*0.04) + current_delta_temp* exp(-2.95*0.05);
			else{
				t[m][n][o]->r->stats.pre_temperature1 =  current_temp + current_delta_temp* exp(-1.98*0.01);
				//if(TEMP_THRESHOLD - t[m][n][o]->r->stats.pre_temperature1>0)
			        //	temp_budget[m][n][o]                  = TEMP_THRESHOLD - t[m][n][o]->r->stats.pre_temperature1; // Jason
				//else
				//	temp_budget[m][n][o]=0;
				//consumption_rate[m][n][o] = t[m][n][o]->r->stats.pre_temperature1 - t[m][n][o]->r->stats.temperature;	
				
				t[m][n][o]->r->stats.pre_temperature2 =  current_temp + current_delta_temp* exp(-1.98*0.01) + current_delta_temp* exp(-1.98*0.02);
				t[m][n][o]->r->stats.pre_temperature3 =  current_temp + current_delta_temp* exp(-1.98*0.01) + current_delta_temp* exp(-1.98*0.02) +
				                                         current_delta_temp* exp(-1.98*0.03);
				t[m][n][o]->r->stats.pre_temperature4 =  current_temp + current_delta_temp* exp(-1.98*0.01) + current_delta_temp* exp(-1.98*0.02) +
				                                         current_delta_temp* exp(-1.98*0.03) + current_delta_temp* exp(-1.98*0.04);
				t[m][n][o]->r->stats.pre_temperature5 =  current_temp + current_delta_temp* exp(-1.98*0.01) + current_delta_temp* exp(-1.98*0.02) +
				                                         current_delta_temp* exp(-1.98*0.03) + current_delta_temp* exp(-1.98*0.04) + 
														 current_delta_temp* exp(-1.98*0.05);
				t[m][n][o]->r->stats.pre_temperature6 =  current_temp + current_delta_temp* exp(-1.98*0.01) + current_delta_temp* exp(-1.98*0.02) +
				                                         current_delta_temp* exp(-1.98*0.03) + current_delta_temp* exp(-1.98*0.04) + 
														 current_delta_temp* exp(-1.98*0.05) + current_delta_temp* exp(-1.98*0.06);														 
			
			}
			
			t[m][n][o]->r->stats.last_pre_temperature1 = t[m][n][o]->r->stats.pre_temperature1;
			t[m][n][o]->r->stats.last_pre_temperature2 = t[m][n][o]->r->stats.pre_temperature2;
			t[m][n][o]->r->stats.last_pre_temperature3 = t[m][n][o]->r->stats.pre_temperature3;
			t[m][n][o]->r->stats.last_pre_temperature4 = t[m][n][o]->r->stats.pre_temperature4;
			t[m][n][o]->r->stats.last_pre_temperature5 = t[m][n][o]->r->stats.pre_temperature5;
			t[m][n][o]->r->stats.last_pre_temperature6 = t[m][n][o]->r->stats.pre_temperature6;		
	        
		}
		else
		{
			t[m][n][o]->r->stats.pre_temperature1 = t[m][n][o]->r->stats.temperature;
			t[m][n][o]->r->stats.pre_temperature2 = t[m][n][o]->r->stats.temperature;
			t[m][n][o]->r->stats.pre_temperature3 = t[m][n][o]->r->stats.temperature;
			t[m][n][o]->r->stats.pre_temperature4 = t[m][n][o]->r->stats.temperature;
			t[m][n][o]->r->stats.pre_temperature5 = t[m][n][o]->r->stats.temperature;
			t[m][n][o]->r->stats.pre_temperature6 = t[m][n][o]->r->stats.temperature;		
		}
	}
	
	
	// Derek 2012.12.10
	// Thermal factor(location prone to be hot spot)

	// for(o=0; o < NoximGlobalParams::mesh_dim_z; o++){
	// 	thermal_factor[0][0][o]=1;   thermal_factor[1][0][o]=2;   thermal_factor[2][0][o]=3;   thermal_factor[3][0][o]=3;
	// 	thermal_factor[4][0][o]=3;   thermal_factor[5][0][o]=2;   thermal_factor[6][0][o]=2;   thermal_factor[7][0][o]=1;		

	// 	thermal_factor[0][1][o]=2;   thermal_factor[1][1][o]=4;   thermal_factor[2][1][o]=5;   thermal_factor[3][1][o]=5;
	// 	thermal_factor[4][1][o]=5;   thermal_factor[5][1][o]=3;   thermal_factor[6][1][o]=2;   thermal_factor[7][1][o]=1;	

	// 	thermal_factor[0][2][o]=3;   thermal_factor[1][2][o]=5;   thermal_factor[2][2][o]=6;   thermal_factor[3][2][o]=7;
	// 	thermal_factor[4][2][o]=6;   thermal_factor[5][2][o]=5;   thermal_factor[6][2][o]=3;   thermal_factor[7][2][o]=2;		
		
	// 	thermal_factor[0][3][o]=4;   thermal_factor[1][3][o]=6;   thermal_factor[2][3][o]=7;   thermal_factor[3][3][o]=8;
	// 	thermal_factor[4][3][o]=7;   thermal_factor[5][3][o]=6;   thermal_factor[6][3][o]=4;   thermal_factor[7][3][o]=2;		

	// 	thermal_factor[0][4][o]=4;   thermal_factor[1][4][o]=6;   thermal_factor[2][4][o]=7;   thermal_factor[3][4][o]=8;
	// 	thermal_factor[4][4][o]=7;   thermal_factor[5][4][o]=6;   thermal_factor[6][4][o]=4;   thermal_factor[7][4][o]=2;		
		
	// 	thermal_factor[0][5][o]=3;   thermal_factor[1][5][o]=5;   thermal_factor[2][5][o]=6;   thermal_factor[3][5][o]=7;
	// 	thermal_factor[4][5][o]=6;   thermal_factor[5][5][o]=5;   thermal_factor[6][5][o]=3;   thermal_factor[7][5][o]=2;		

	// 	thermal_factor[0][6][o]=2;   thermal_factor[1][6][o]=4;   thermal_factor[2][6][o]=5;   thermal_factor[3][6][o]=5;
	// 	thermal_factor[4][6][o]=5;   thermal_factor[5][6][o]=4;   thermal_factor[6][6][o]=2;   thermal_factor[7][6][o]=1;		

	// 	thermal_factor[0][7][o]=1;   thermal_factor[1][7][o]=2;   thermal_factor[2][7][o]=3;   thermal_factor[3][7][o]=3;
	// 	thermal_factor[4][7][o]=3;   thermal_factor[5][7][o]=2;   thermal_factor[6][7][o]=2;   thermal_factor[7][7][o]=1;			
	// }

	// Derek 2012.12.17
	// Penalty Factor(decline largely when temperature close to limit)

	// for(o=0; o < NoximGlobalParams::mesh_dim_z; o++)
	// for(n=0; n < NoximGlobalParams::mesh_dim_y; n++) 
	// for(m=0; m < NoximGlobalParams::mesh_dim_x; m++) {	
	// 	if( (t[m][n][o]->r->stats.pre_temperature1)<90 )
	// 		penalty_factor[m][n][o]= 1- 0.1*(t[m][n][o]->r->stats.pre_temperature1-85);
	// 	else if((t[m][n][o]->r->stats.pre_temperature1)>=90 && (t[m][n][o]->r->stats.pre_temperature1)<95)
	// 		penalty_factor[m][n][o]= 0.5- 0.04*(t[m][n][o]->r->stats.pre_temperature1-90);
	// 	else if((t[m][n][o]->r->stats.pre_temperature1)>=95 && (t[m][n][o]->r->stats.pre_temperature1)<100)
	// 		penalty_factor[m][n][o]= 0.3- 0.06*(t[m][n][o]->r->stats.pre_temperature1-95);
	// 	else
	// 		penalty_factor[m][n][o]= 0;
	// }

	
	// Total Thermal Budget
	// for(o=0; o < NoximGlobalParams::mesh_dim_z; o++)
	// for(n=0; n < NoximGlobalParams::mesh_dim_y; n++) 
	// for(m=0; m < NoximGlobalParams::mesh_dim_x; m++) {	
	// 	if(consumption_rate[m][n][o]>0){
	// 	//	t[m][n][o]->r->TBDB(consumption_rate[m][n][o]);
	// 		MTTT[m][n][o] = (temp_budget[m][n][o]/**penalty_factor[m][n][o]*/)/consumption_rate[m][n][o];	
	// 	}
	// }


}

void NoximNoC::PRETempLog(){
	int o,n,m,idx;

	pretemp_file<<"Cycletime: "<<getCurrentCycleNum()<<"\n";
	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++){
		pretemp_file<<"XY"<<o<<"=[\n";
		for(n=NoximGlobalParams::mesh_dim_y-1; n > -1; n--){
			for(m=0; m < NoximGlobalParams::mesh_dim_x; m++){
				pretemp_file<< t[m][n][o]->r->stats.pre_temperature1 << "\t";
			}
			pretemp_file<<"\n";
		}
		pretemp_file<<"]\n"<<"\n";
	}
	pretemp_file.flush();
	
	throt_level<<"Cycletime: "<<getCurrentCycleNum()<<"\n";
	for(o=0; o < NoximGlobalParams::mesh_dim_z; o++){
		throt_level<<"XY"<<o<<"=[\n";
		for(n=NoximGlobalParams::mesh_dim_y-1; n > -1; n--){
			for(m=0; m < NoximGlobalParams::mesh_dim_x; m++){
				throt_level<< throttling2[m][n][o] << "\t";
			}
			throt_level<<"\n";
		}
		throt_level<<"]\n"<<"\n";
	}
	throt_level.flush();
}

bool NoximNoC::EmergencyDecision()
{	
	bool isEmergency = false;
	if     (NoximGlobalParams::throt_type == THROT_GLOBAL)
		GlobalThrottle(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_DISTRIBUTED)
		DistributedThrottle(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_FGR)
		FGR(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_PREDICTION)
		PredictionThrottle(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_TAVT)
		TAVT(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_TAVT_MAX)
		TAVT_MAX(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_VERTICAL)
		Vertical(isEmergency);
	else if(NoximGlobalParams::throt_type == THROT_VERTICAL_MAX)
		Vertical_MAX(isEmergency);
	else return isEmergency;//THROT_NORMAL,THROT_TEST do nothing, because the topology won't change
	return isEmergency;
}
void NoximNoC::GlobalThrottle(bool& isEmergency){
	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
		if(t[x][y][z]->r->stats.temperature > NoximGlobalParams::threshold_para){ // each temperature of routers exceed temperature threshould
			isEmergency = true;
			break;
	}
	if(isEmergency){
	    for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
	    for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
	    for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++){
	    	t[x][y][z]->pe->IntoEmergency();
	    	t[x][y][z]->r ->IntoEmergency();
	    	throttling[x][y][z] = isEmergency; 
	    }
	}
	else{
		for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
	    for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
	    for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++){
	    	t[x][y][z]->pe->OutOfEmergency();
	    	t[x][y][z]->r ->OutOfEmergency();
	    	throttling[x][y][z] = isEmergency; 
	    }
	}
}
void NoximNoC::DistributedThrottle(bool& isEmergency){
	for(int z=0; z < NoximGlobalParams::mesh_dim_z     ; z++)
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++){
		NoximCoord coord;
        coord.x = x;
        coord.y = y;
		coord.z = z;
		int node_id = best_grid[z][y][x];
        //int node_id = coord2Id(coord);
		if(node_id%NoximGlobalParams::classification==rema){
			if(t[x][y][z]->r->stats.temperature > NoximGlobalParams::threshold_para ){
				//isEmergency = true;
				throttling2[x][y][z] = 4;
				throttling[x][y][z] = true;
				isEmergency = true;
				t[x][y][z]->pe->IntoEmergency();
				t[x][y][z]->r ->IntoEmergency();
			}
			else{
				t[x][y][z]->pe->OutOfEmergency();
				t[x][y][z]->r ->OutOfEmergency();
				throttling2[x][y][z] = 0;
				throttling[x][y][z] = false;
			}
		}
	}
}
//New Fine-Grained throttling, still in test
void NoximNoC::FGR(bool& isEmergency){
	for(int z=0; z < NoximGlobalParams::mesh_dim_z     ; z++)
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++){
		float criteria = deltaT[x][y][z] / 2;
		NoximCoord coord;
        coord.x = x;
        coord.y = y;
		coord.z = z;
		int node_id = best_grid[z][y][x];
        //int node_id = coord2Id(coord);
		if(node_id%NoximGlobalParams::classification==rema){
			if(t[x][y][z]->r->stats.temperature > NoximGlobalParams::threshold_para - criteria ){
				//isEmergency = true;
				double diff;
				diff = t[x][y][z]->r->stats.temperature - NoximGlobalParams::threshold_para + criteria;
				if(diff<criteria)
					throttling2[x][y][z] = 1;
				else if(diff<criteria*2)
					throttling2[x][y][z] = 2;
				else if(diff<criteria*3)
					throttling2[x][y][z] = 3;
				else
					throttling2[x][y][z] = 4;
				
				throttling[x][y][z] = true;
				isEmergency = true;
				t[x][y][z]->pe->IntoEmergency();
				t[x][y][z]->r ->IntoEmergency();
			}
			else{
				t[x][y][z]->pe->OutOfEmergency();
				t[x][y][z]->r ->OutOfEmergency();
				throttling2[x][y][z] = 0;
				throttling[x][y][z] = false;
			}
		}
	}
}
void NoximNoC::PredictionThrottle(bool& isEmergency){
	bool flag=0;
	for(int z=0; z < NoximGlobalParams::mesh_dim_z  ; z++)
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++){
		float criteria = deltaT[x][y][z] / 2;
		NoximCoord coord;
        coord.x = x;
        coord.y = y;
		coord.z = z;
        int node_id = coord2Id(coord);
		if(node_id%NoximGlobalParams::classification==rema){

			bool flag_neighbor_no_throt=0;
			bool a,c,d,e,f;
			c = (x==0)?1:(t[x-1][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para);
			d = (x==NoximGlobalParams::mesh_dim_x-1)?1:(t[x+1][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para);
			e = (y==0)?1:(t[x][y-1][z]->r->stats.temperature < NoximGlobalParams::threshold_para);
			f = (y==NoximGlobalParams::mesh_dim_y-1)?1:(t[x][y+1][z]->r->stats.temperature < NoximGlobalParams::threshold_para);
			a = (t[x][y][z]->r->stats.temperature > NoximGlobalParams::threshold_para-0.5);

			if(a&c&d&e&f){
				flag_neighbor_no_throt=1;
			}

			if(t[x][y][z]->r->stats.throt_d>0 ){	
				//cout<<getCurrentCycleNum()<<": in["<<x<<"]["<<y<<"]["<<z<<"], throt YES,  throt_d: "<<t[x][y][z]->r->stats.throt_d<<endl;
				isEmergency = true;
				throttling2[x][y][z] = t[x][y][z]->r->stats.throt_d;
				t[x][y][z]->pe->IntoEmergency();
				t[x][y][z]->r ->IntoEmergency();
			}
			else if(t[x][y][z]->r->stats.temperature > NoximGlobalParams::threshold_para){
				//isEmergency = true;
				double diff;
				diff = t[x][y][z]->r->stats.temperature - NoximGlobalParams::threshold_para + criteria;
				if(diff<criteria)
					throttling2[x][y][z] = 1;
				else if(diff<criteria*2)
					throttling2[x][y][z] = 2;
				else if(diff<criteria*3)
					throttling2[x][y][z] = 3;
				else
					throttling2[x][y][z] = 4;

				isEmergency = true;
				t[x][y][z]->pe->IntoEmergency();
				t[x][y][z]->r ->IntoEmergency();
			}
			else if(flag_neighbor_no_throt){
				isEmergency = true;
				throttling2[x][y][z] = 1;
				t[x][y][z]->pe->IntoEmergency();
				t[x][y][z]->r ->IntoEmergency();
			}
			else{	
				//cout<<getCurrentCycleNum()<<": in["<<x<<"]["<<y<<"]["<<z<<"], throt NO,  throt_d: "<<t[x][y][z]->r->stats.throt_d<<endl;
				t[x][y][z]->pe->OutOfEmergency();	
				t[x][y][z]->r ->OutOfEmergency();
				throttling2[x][y][z] = 0; 
			}
		}
	}
}
void NoximNoC::TAVT(bool& isEmergency){
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++ )
	for(int z=0; z < NoximGlobalParams::mesh_dim_z - 1 ; z++ )
	{
		if (t[x][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para ){	
			t[x][y][z]->pe->OutOfEmergency();		
			t[x][y][z]->r ->OutOfEmergency();
			throttling[x][y][z] = 0; 
			if( t[x][y][z]->r->stats.temperature > 0/*NoximGlobalParams::beltway_trigger*/ && NoximGlobalParams::beltway )beltway[x][y][z]       = true;
			else	beltway[x][y][z]       = false;
		}
		else{ // >TEMP_THRESHOLD
			isEmergency = true;
			for( int zz = 0 ; zz < z + 1 ; zz++){
			// for( int zz = 0 ; zz < NoximGlobalParams::mesh_dim_z ; zz++){
				if(zz < NoximGlobalParams::mesh_dim_z-1){	//Bottom chip layer won't be throttle
					t[x][y][zz]->pe->IntoEmergency();
					t[x][y][zz]->r ->IntoEmergency();
					throttling[x][y][zz]       = 1;
				}
			}
			break;
		}
	}
	Reconfiguration();
}

void NoximNoC::TAVT_MAX(bool& isEmergency){
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++ )
	for(int z=0; z < NoximGlobalParams::mesh_dim_z     ; z++ )
	{
		if (t[x][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para ){	
			t[x][y][z]->pe->OutOfEmergency();		
			t[x][y][z]->r ->OutOfEmergency();
			throttling[x][y][z] = 0; 
			if( t[x][y][z]->r->stats.temperature > NoximGlobalParams::beltway_trigger && NoximGlobalParams::beltway )beltway[x][y][z]       = true;
			else                                                                					beltway[x][y][z]       = false;
		}
		else{ // >TEMP_THRESHOLD
			isEmergency = true;
			for( int zz = 0 ; zz < z + 1 ; zz++){
					t[x][y][zz]->pe->IntoEmergency();
					t[x][y][zz]->r ->IntoEmergency();
					throttling[x][y][zz]       = 1;
			}
			break;
		}
	}
	Reconfiguration();	
}

void NoximNoC::Vertical(bool& isEmergency){
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++ )
	for(int z=0; z < NoximGlobalParams::mesh_dim_z - 1 ; z++ )
	{
		if (t[x][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para ){	
			t[x][y][z]->pe->OutOfEmergency();		
			t[x][y][z]->r ->OutOfEmergency();
			throttling[x][y][z] = 0; 
			if( t[x][y][z]->r->stats.temperature > NoximGlobalParams::beltway_trigger && NoximGlobalParams::beltway )
				beltway[x][y][z] = true;
			else                                                                					
				beltway[x][y][z] = false;
		}
		else{ // >TEMP_THRESHOLD
			isEmergency = true;
			for( int zz = 0 ; zz < z + 1 ; zz++){
				if(zz < NoximGlobalParams::mesh_dim_z-1){	//Bottom chip layer won't be throttle
					t[x][y][zz]->pe->IntoEmergency();
					t[x][y][zz]->r ->IntoEmergency();
					throttling[x][y][zz]       = 1;
				}
			}
			break;
		}
	}
	Reconfiguration();
}

void NoximNoC::Vertical_MAX(bool& isEmergency){
	for(int y=0; y < NoximGlobalParams::mesh_dim_y     ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x     ; x++ )
	for(int z=0; z < NoximGlobalParams::mesh_dim_z     ; z++ ){
		if (t[x][y][z]->r->stats.temperature < NoximGlobalParams::threshold_para ){	
			t[x][y][z]->pe->OutOfEmergency();		
			t[x][y][z]->r ->OutOfEmergency();
			throttling[x][y][z] = 0; 
			if( t[x][y][z]->r->stats.temperature > NoximGlobalParams::beltway_trigger && NoximGlobalParams::beltway )beltway[x][y][z]       = true;
			else                                                                					beltway[x][y][z]       = false;
		}
		else{ // >TEMP_THRESHOLD
			isEmergency = true;
			for( int zz = 0 ; zz < NoximGlobalParams::mesh_dim_z ; zz++){
					t[x][y][zz]->pe->IntoEmergency();
					t[x][y][zz]->r ->IntoEmergency();
					throttling[x][y][zz]       = 1;
			}
			break;
		}
	}
	Reconfiguration();
}

void NoximNoC::setCleanStage(){
	cout<<getCurrentCycleNum()<<":Into Clean Stage"<<endl;
	for(int z=0; z < NoximGlobalParams::mesh_dim_z ; z++ )//from top to down
	for(int y=0; y < NoximGlobalParams::mesh_dim_y ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x ; x++ ){
		NoximCoord coord;
        coord.x = x;
        coord.y = y;
		coord.z = z;
		int node_id = best_grid[z][y][x];
		if(node_id%NoximGlobalParams::classification==rema){
			t[x][y][z]->pe->IntoCleanStage();
			t[x][y][z]->pe->OutOfEmergency();
			t[x][y][z]->r ->OutOfEmergency();
			throttling[x][y][z]             = false;
		}
	}
}

void NoximNoC::EndCleanStage(){
	cout<<getCurrentCycleNum()<<":Out of Clean Stage"<<endl;
	for(int z=0; z < NoximGlobalParams::mesh_dim_z ; z++ )//from top to down
	for(int y=0; y < NoximGlobalParams::mesh_dim_y ; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x ; x++ ){
		NoximCoord coord;
        coord.x = x;
        coord.y = y;
		coord.z = z;
		int node_id = best_grid[z][y][x];
		if(node_id%NoximGlobalParams::classification==rema){
			t[x][y][z]->pe->OutOfCleanStage();
		}
	}
}

void NoximNoC::findNonXLayer(int &non_throt_layer, int &non_beltway_layer){
	int m, n, layer;
	bool index_throt = false,index_beltway = false;
	non_throt_layer = 0,non_beltway_layer = 0;
	for(layer=NoximGlobalParams::mesh_dim_z - 1 ; layer > -1 ; layer-- ){
		for(n=0; n < NoximGlobalParams::mesh_dim_y; n++) 
		for(m=0; m < NoximGlobalParams::mesh_dim_x; m++)
			index_beltway |= beltway[m][n][layer];
		if (index_beltway){
			non_beltway_layer = layer + 1;
			break;
		}
	}
	for(layer = NoximGlobalParams::mesh_dim_z - 1 ; layer > -1 ; layer-- ){
		for(n=0; n < NoximGlobalParams::mesh_dim_y; n++) 
		for(m=0; m < NoximGlobalParams::mesh_dim_x; m++)
			index_throt   |= throttling[m][n][layer];
		if (index_throt){
			non_throt_layer = layer + 1;
			break;
		}
	}
	assert( non_throt_layer > -1 && non_beltway_layer > -1 );
}


void NoximNoC::calROC(int &col_max, int &col_min, int &row_max, int &row_min,int non_beltway_layer){
	int X_min = 0;
	int X_max = NoximGlobalParams::mesh_dim_x - 1;
	int Y_min = 0;
	int Y_max = NoximGlobalParams::mesh_dim_x - 1;
	int Z     = non_beltway_layer;
	
	int m, n, layer;
	bool index = false;
	for( n = 0 ; n < NoximGlobalParams::mesh_dim_y ; n++ ){
		for(layer=0; layer < Z; layer++) 
		for(m=0; m < NoximGlobalParams::mesh_dim_x; m++)
			index |= beltway[m][n][layer];
		if (index){
			Y_min = n ;
			break;
		}
	}
	index = 0;
	for( n = NoximGlobalParams::mesh_dim_y - 1 ; n > -1  ; n-- ){
		for(layer=0; layer < Z; layer++) 
		for(m=0; m < NoximGlobalParams::mesh_dim_x; m++)
			index |= beltway[m][n][layer];
		if (index){
			Y_max = n ;
			break;
		}
	}
	index = 0;
	for( m = 0 ; m < NoximGlobalParams::mesh_dim_x ; m++ ){
		for(layer=0; layer < Z; layer++) 
		for(n = Y_min ; n < Y_max + 1 ; n++)
			index |= beltway[m][n][layer];
		if (index){
			X_min = m ;
			break;
		}
	}
	index = 0;
	for( m = NoximGlobalParams::mesh_dim_x - 1 ; m > -1  ; m-- ){
		for(layer = 0; layer < Z; layer++) 
		for(n=Y_min ; n < Y_max + 1; n++){
			index |= beltway[m][n][layer];
			}
		if (index){
			X_max = m ;
			break;
		}
	}
	col_min = X_min;
	col_max = X_max;
	row_min = Y_min;
	row_max = Y_max;
}

void NoximNoC::Reconfiguration(){
	int non_beltway_layer,non_throt_layer;
	int col_max,col_min,row_max,row_min;
	findNonXLayer(non_throt_layer,non_beltway_layer); 
	calROC(col_max,col_min,row_max,row_min,non_beltway_layer);
	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++ )
	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++ ) 
	for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++ )
		t[x][y][z]->pe->RTM_set_var(non_throt_layer, non_beltway_layer, col_max, col_min, row_max, row_min);
	transient_topology<<"non_beltway_layer,non_throt_layer = "<<non_beltway_layer<<","<<non_throt_layer<<endl;
	transient_topology<<"col_max,col_min,row_max,row_min = "<<col_max<<","<<col_min<<","<<row_max<<","<<row_min<<endl;
}
bool NoximNoC::_equal(int x, int y, int z, int m, int n, int o){
	return ( x == m )&&( y == n )&&( z == o );
}

// bool NoximNoC::_CleanDone(){
// 	bool clean = true;
// 	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++)
// 	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++)
// 	for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++){
	
// 		if( !t[x][y][z]->pe->flit_queue.empty() ){
// 			cout<<"t["<<x<<"]["<<y<<"]["<<z<<"] pe flit_queue is not empty "<<t[x][y][z]->pe->flit_queue.front()<<endl;
// 		}
// 		if( !t[x][y][z]->pe->packet_queue.empty() ){
// 			cout<<"t["<<x<<"]["<<y<<"]["<<z<<"] pe packet_queue is not empty "<<t[x][y][z]->pe->packet_queue.front().dst_id<<endl;
// 		}
// 		for(int d = 0; d < DIRECTIONS+1; d++){
// 			if ( !(t[x][y][z]->r->buffer[d].IsEmpty()) ){
// 				clean = false;
				
// 				if( ((int)(t[x][y][z]->r->buffer[d].Front().timestamp) % (int) (TEMP_REPORT_PERIOD) ) >  NoximGlobalParams::clean_stage_time){
// 					int output_channel = t[x][y][z]->r->getFlitRoute(d);
// 					cout<<"In node t["<<x<<"]["<<y<<"]["<<z<<"] direction "<<d<<" waiting time "<<getCurrentCycleNum() - t[x][y][z]->r->buffer[d].Front().waiting_cnt<<" ";
// 					cout<<t[x][y][z]->r->buffer[d].Front()<<" ";
// 					cout<<"This flit is route to "<<output_channel<<" ack("<<t[x][y][z]->r->ack_tx[output_channel]<<") avalible("<<t[x][y][z]->r->getDirAvailable(output_channel)<<")"<<endl;
// 				}
// 			}
// 		}
// 	}
// 	return clean;
// }

void NoximNoC::TransientLog(){
	//calculate the period throughtput

	int packet_in_buffer           =0;
	int throttle_num               =0;
	float max_temp                 =0;
	int total_transmit             =0;
	int total_adaptive_transmit    =0;
	int total_dor_transmit         =0;
	int total_dw_transmit          =0;
	int total_mid_adaptive_transmit=0;
	int total_mid_dor_transmit     =0;
	int total_mid_dw_transmit      =0;
	int total_beltway              =0;
	int max_delay                  =0;
	int max_delay_id               =0;
	int max_delay_id_d             =0;
	
		
	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++)
	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++)
	for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++){
		if( !t[x][y][z]->pe->flit_queue.empty() ){
			cout<<"t["<<x<<"]["<<y<<"]["<<z<<"] pe flit_queue is not empty "<<t[x][y][z]->pe->flit_queue.front()<<", flit_queue size "<<t[x][y][z]->pe->flit_queue.size()<<endl;
		}
		
		if( !t[x][y][z]->pe->packet_queue.empty() ){
			cout<<"t["<<x<<"]["<<y<<"]["<<z<<"] pe packet_queue is not empty "<<t[x][y][z]->pe->packet_queue.front().dst_id<<", packet_queue size"<<t[x][y][z]->pe->packet_queue.size()<<endl;
		}
		
		for(int d = 0; d < DIRECTIONS+1; d++){
			if ( !(t[x][y][z]->r->buffer[d].IsEmpty()) ){
				packet_in_buffer++;
				int output_channel = t[x][y][z]->r->getFlitRoute(d);
				//if( _emergency ){
				cout<<"In node t["<<x<<"]["<<y<<"]["<<z<<"] direction "<<d<<" waiting time "<<getCurrentCycleNum() - t[x][y][z]->r->buffer[d].Front().waiting_cnt<<" ";
				cout<<t[x][y][z]->r->buffer[d].Front()<<" ";
				cout<<"This flit is route to "<<output_channel<<" ack("<<t[x][y][z]->r->ack_tx[output_channel]<<") avalible("<<t[x][y][z]->r->getDirAvailable(output_channel)<<")"<<endl;
				cout<<"Reservation table:"<<d<<" route to "<<t[x][y][z]->r->reservation_table.getOutputPort(d)<<" ";
				cout<<"beltway: "<<t[x][y][z]->r->buffer[d].Front().beltway<<" ";
				cout<<"hop no.: "<<t[x][y][z]->r->buffer[d].Front().hop_no<<" ";
				cout<<"arr mid: "<<t[x][y][z]->r->buffer[d].Front().arr_mid<<" ";
				cout<<"arr mid: "<<t[x][y][z]->r->buffer[d].Front().arr_mid<<" ";
				cout<<"DW_layer: "<<t[x][y][z]->r->buffer[d].Front().DW_layer<<" ";
				cout<<endl;
				if( ((int)(t[x][y][z]->r->buffer[d].Front().timestamp) % (int) (TEMP_REPORT_PERIOD) ) > (int) (TEMP_REPORT_PERIOD) - NoximGlobalParams::clean_stage_time){
					//cout<<"flit timestamp = "<<t[x][y][z]->r->buffer[d].Front().timestamp<<endl;
					//assert(false);
				}
					//assert(false);
				//}
			}
		
			if( d == 0){
				if( throttling[x][y][z] )throttle_num++;
				max_temp = ( t[x][y][z]->r->stats.temperature > max_temp)?t[x][y][z]->r->stats.temperature:max_temp;
				// total_transmit             += t[x][y][z]->pe->getTransient_Total_Transmit();
				total_adaptive_transmit    += t[x][y][z]->pe->getTransient_Adaptive_Transmit();
				total_dor_transmit         += t[x][y][z]->pe->getTransient_DOR_Transmit();
				total_dw_transmit          += t[x][y][z]->pe->getTransient_DW_Transmit();
				total_mid_adaptive_transmit+= t[x][y][z]->pe->getTransient_Mid_Adaptive_Transmit();
				total_mid_dor_transmit     += t[x][y][z]->pe->getTransient_Mid_DOR_Transmit();
				total_mid_dw_transmit      += t[x][y][z]->pe->getTransient_Mid_DW_Transmit();
				total_beltway              += t[x][y][z]->pe->getTransient_Beltway_Transmit();
			}
		}
		t[x][y][z]->pe->ResetTransient_Transmit();
	}
	total_transmit = total_adaptive_transmit    + 
					total_dor_transmit         +       
					total_dw_transmit          +       
					total_mid_adaptive_transmit+
					total_mid_dor_transmit     +
					total_mid_dw_transmit      +
					total_beltway              ;	
		
	transient_log_throughput<<getCurrentCycleNum()<<"\t"
	                        <<total_transmit<<"\t\t\t"
							<<total_transmit/TEMP_REPORT_PERIOD<<"\t\t"
							<<throttle_num<<"\t\t\t\t"
							<<max_temp<<"\t\t"
							<<total_adaptive_transmit    <<"\t\t"
							<<total_dor_transmit         <<"\t\t"
							<<total_dw_transmit          <<"\t"
							<<total_mid_adaptive_transmit<<"\t\t\t\t"
							<<total_mid_dor_transmit     <<"\t\t\t"
							<<total_mid_dw_transmit      <<"\t\t"
							<<total_beltway              <<"\t"
							<<endl;
	transient_topology<<"Throttling Table @"<<getCurrentCycleNum()<<" "<<throttle_num<<" nodes are throttled, "<<packet_in_buffer<<" Non-Empty Buffers, Throughput "
				    <<total_transmit/TEMP_REPORT_PERIOD<<endl;
	for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++)
		transient_topology<<"Layer "<<z<<"\t";
	transient_topology<<"\n";

	for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++){
		for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++){
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++){
				if(throttling[x][y][z])transient_topology<<"X";
				else if( beltway[x][y][z] )transient_topology<<"*";
				else transient_topology<<".";
			}
			transient_topology<<"\t";	
		}
		transient_topology<<endl;			
	}
}

void NoximNoC::_setThrot(int i, int j, int k){
	throttling2[i][j][k] = 3; //@ct: for new FGP test
	throttling[i][j][k] = 1;
	t[i][j][k]->pe->IntoEmergency();
	t[i][j][k]->r ->IntoEmergency();
}

void NoximNoC::_setNormal(int i, int j, int k){
	throttling[i][j][k] = 0;
	t[i][j][k]->pe->OutOfEmergency();
	t[i][j][k]->r ->OutOfEmergency();
}

void NoximNoC::_throt_case_setting( int throt_case ){
	switch( throt_case ){
	case 1 ://normal
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++)
			_setNormal(i,j,k);
	break;
	case 2 :
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ..X.....        ..X.....        ........        ........
		// ..X.....        ..X.....        ........        ........
		// ....X...        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,2,0,i,j,k) || _equal(2,3,0,i,j,k) || _equal(4,4,0,i,j,k) ||
				_equal(2,2,1,i,j,k) || _equal(2,3,1,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 3 :
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ..X.....        ..X.....        ........        ........
		// ..X.....        ..X.....        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,2,0,i,j,k) || _equal(2,3,0,i,j,k) || 
				_equal(2,2,1,i,j,k) || _equal(2,3,1,i,j,k) ||
				_equal(2,2,2,i,j,k) || _equal(2,3,2,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 4 :
		// ........        ........        ........        ........
		// ........        ........        ........        ........        
		// ..X.....        ..X.....        ........        ........        
		// ..X.....        ..X.....        ........        ........        
		// ........        ........        ........        ........        
		// ........        ........        ........        ........        
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,2,0,i,j,k) || _equal(2,3,0,i,j,k) ||
				_equal(2,2,1,i,j,k) || _equal(2,3,1,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 5:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
		if(k < (NoximGlobalParams::mesh_dim_z - 1)){
				if( ((i == 1)&&(j == 1)) || ((i == 1)&&(j == 2)) || ((i == 2)&&(j == 2)) || ((i == 2)&&(j == 1))
				|| ((i == 6)&&(j == 5)) || ((i == 5)&&(j == 6)) || ((i == 6)&&(j == 6)) || ((i == 5)&&(j == 5)) ){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 6 :
		// ........        ........        ........        ........
		// ....x...        ........        ........        ........
		// ....x...        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,5,0,i,j,k) || _equal(2,6,0,i,j,k) ||
				_equal(3,5,0,i,j,k) || _equal(3,6,0,i,j,k) || 
				_equal(4,1,0,i,j,k) || _equal(4,2,0,i,j,k) || 
				_equal(2,5,1,i,j,k) || _equal(2,6,1,i,j,k) ||  
				_equal(3,5,1,i,j,k) || _equal(3,6,1,i,j,k)   )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 7 :
		// ........        ........        ........        ........
		// ....x...        ....x...        ........        ........
		// ....x...        ....x...        ........        ........
		// ........        ........        ........        ........
		// .xxxx...        ..x.....        ........        ........
		// ..xx....        ..xx....        ..x.....        ........
		// ..xx....        ..xx....        ..x.....        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(1,4,0,i,j,k) ||  
				_equal(2,4,0,i,j,k) || _equal(2,5,0,i,j,k) || _equal(2,6,0,i,j,k) || 
				_equal(3,4,0,i,j,k) || _equal(3,5,0,i,j,k) || _equal(3,6,0,i,j,k) || 
				_equal(4,1,0,i,j,k) || _equal(4,2,0,i,j,k) || _equal(4,4,0,i,j,k) ||
				_equal(2,4,1,i,j,k) || _equal(2,5,1,i,j,k) || _equal(2,6,1,i,j,k) || 
				_equal(3,5,1,i,j,k) || _equal(3,6,1,i,j,k) || 
				_equal(4,1,1,i,j,k) || _equal(4,2,1,i,j,k) || 
				_equal(2,5,2,i,j,k) || _equal(2,6,2,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 8 :
		// ........        ........        ........        ........
		// ....x...        ........        ........        ........
		// ....x...        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,5,0,i,j,k) || _equal(2,6,0,i,j,k) || _equal(3,5,0,i,j,k) || 
				_equal(3,6,0,i,j,k) || _equal(4,1,0,i,j,k) || _equal(4,2,0,i,j,k) || 
				_equal(2,5,1,i,j,k) || _equal(2,6,1,i,j,k) || _equal(3,5,1,i,j,k) || 
				_equal(3,6,1,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 9 :
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ....x...        ........        ........        ........
		// ........        ........        ........        ........
		// ..xx....        ........        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,4,0,i,j,k) || _equal(2,5,0,i,j,k) || _equal(3,4,0,i,j,k) || 
				_equal(3,5,0,i,j,k) || _equal(4,2,0,i,j,k) || _equal(2,5,1,i,j,k) || 
				_equal(3,5,1,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 10:
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ..xx....        ........        ........        ........
		// ..xx....        ..xx....        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		// ........        ........        ........        ........
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if( _equal(2,4,0,i,j,k) || _equal(2,5,0,i,j,k) || _equal(3,4,0,i,j,k) || 
				_equal(3,5,0,i,j,k) || _equal(2,5,1,i,j,k) || _equal(3,5,1,i,j,k) )
				_setThrot(i,j,k);
			else
				_setNormal(i,j,k);
		}
	break;
	case 11:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
		if(k < (NoximGlobalParams::mesh_dim_z - 2)){
				if( ((i == 1)&&(j == 1)) || ((i == 1)&&(j == 2)) || ((i == 2)&&(j == 2)) || ((i == 2)&&(j == 1))
				|| ((i == 6)&&(j == 5)) || ((i == 5)&&(j == 6)) || ((i == 6)&&(j == 6)) || ((i == 5)&&(j == 5)) ){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 12:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if(k < (NoximGlobalParams::mesh_dim_z - 2)){
				if(((i == 3))&&((j == 3))){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 13:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
		if(k < (NoximGlobalParams::mesh_dim_z - 1)){
				if( ((i == 1)&&(j == 5)) || ((i == 1)&&(j == 6)) || ((i == 2)&&(j == 5)) || ((i == 2)&&(j == 6))
				|| ((i == 5)&&(j == 1)) || ((i == 6)&&(j == 1)) || ((i == 5)&&(j == 2)) || ((i == 6)&&(j == 2)) ){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 14:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
			if(k < (NoximGlobalParams::mesh_dim_z - 1)){
				if(((i == 3))&&((j == 3))){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 15:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
		if(k < (NoximGlobalParams::mesh_dim_z - 1)){
				if( ((i == 4)&&(j == 4)) || ((i == 4)&&(j == 3)) || ((i == 3)&&(j == 4)) || ((i == 3)&&(j == 3)) ){
				//if( ((i == 1)&&(j == 1)) || ((i == 6)&&(j == 6)) ){
				//if ( false ){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
	case 16:
		for (int k=0; k<NoximGlobalParams::mesh_dim_z; k++)
		for (int j=0; j<NoximGlobalParams::mesh_dim_y; j++)
		for (int i=0; i<NoximGlobalParams::mesh_dim_x; i++){
		if(k < (NoximGlobalParams::mesh_dim_z - 1)){
				if( ((i == 1)&&(j == 1)) || ((i == 6)&&(j == 6)) ){
					throttling[i][j][k] = 1;
					t[i][j][k]->pe->IntoEmergency();
					t[i][j][k]->r ->IntoEmergency();
				}
				else{
					throttling[i][j][k] = 0;
					t[i][j][k]->pe->OutOfEmergency();
					t[i][j][k]->r ->OutOfEmergency();
				}				
			}
			else {
				throttling[i][j][k] = 0;
				beltway[i][j][k]    = false;
				t[i][j][k]->pe->OutOfEmergency();
				t[i][j][k]->r ->OutOfEmergency();
			}
		}
	break;
		
	} 
}
