// stock-api.cpp : DLL 用にエクスポートされる関数を定義します。
//
#include "pch.h"
#include "framework.h"
#include <cpprest/http_client.h>
#include <locale.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace std;

double _predict(const wchar_t* ws_endpoint, const wchar_t* ws_model_name, double* array, int arrayNumber, const wchar_t* ws_timestamp, const wchar_t* ws_expname, int currentBar, int NetProfit, int OpenPositionProfit)
{
	json::value postData;
	json::value feature = json::value::array();
	for (int i = 0; i < arrayNumber; i++) {
		feature[i] = array[i];
	}
	postData[L"model"] = json::value::string(ws_model_name);
	postData[L"feature"] = feature;
	postData[L"timestamp"] = json::value::string(ws_timestamp);
	postData[L"expname"] = json::value::string(ws_expname);
	postData[L"currentBar"] = currentBar;
	postData[L"NetProfit"] = NetProfit;
	postData[L"OpenPositionProfit"] = OpenPositionProfit;
	try
	{
		http_client client(ws_endpoint);
		auto task{ client.request(methods::POST, L"", postData.serialize(), L"application/json") };
		return task.then([ws_model_name](http_response resp) {
			if (resp.status_code() == status_codes::OK)
			{
				return resp.extract_json();
			}
			}).then([ws_model_name](json::value json) {
				// 応答結果を返す
				if (json[L"status"].as_integer() == 0) {
					return json[L"prediction"][0].as_double();
				}
				else {
					wofstream outputfile("C:\\stock-api-error.log", ios_base::out | ios_base::app);
					outputfile << "symbolname:" << ws_model_name << " status:" << json[L"status"].as_integer() << " message:" << json[L"message"].as_string() << endl;
					outputfile.close();
				}
				return 0.0;
			}).get();
	}
	catch (const std::exception& e)
	{
		wofstream outputfile("C:\\stock-api-error.log", ios_base::out | ios_base::app);
		outputfile  << "model_name:" << ws_model_name << " predict error:" << e.what() << endl;
		outputfile.close();
		return 0;
	}
}

int _backward(const wchar_t* ws_endpoint, const wchar_t* ws_model_name, double reward)
{
	json::value postData;
	postData[L"model"] = json::value::string(ws_model_name);
	postData[L"reward"] = reward;
	try
	{
		http_client client(ws_endpoint);
		auto task{ client.request(methods::POST, L"", postData.serialize(), L"application/json") };
		return task.then([ws_model_name](http_response resp) {
			if (resp.status_code() == status_codes::OK)
			{
				return resp.extract_json();
			}
			}).then([ws_model_name](json::value json) {
				// 応答結果を返す
				if (json[L"status"].as_integer() == 0) {
					return 0;
				}
				return 0;
			}).get();
	}
	catch (const std::exception& e)
	{
		wofstream outputfile("C:\\stock-api-error.log", ios_base::out | ios_base::app);
		outputfile << "model_name:" << ws_model_name << " backward error:" << e.what() << endl;
		outputfile.close();
		return 0;
	}
}

// endpoint: http://192.168.11.30:8000/predict
// feature value: v1 - v11
extern "C" double __declspec(dllexport) __stdcall predict(
	const char* endpoint,
	const char* model_name,
	const char* timestamp,
	const char* expname,
	int currentBar,
	int NetProfit,
	int OpenPositionProfit,
	double v1,
	double v2,
	double v3,
	double v4,
	double v5,
	double v6,
	double v7,
	double v8,
	double v9,
	double v10,
	double v11,
	double v12,
	double v13,
	double v14,
	double v15,
	double v16,
	double v17,
	double v18,
	double v19,
	double v20,
	double v21,
	double v22,
	double v23,
	double v24,
	double v25,
	double v26,
	double v27,
	double v28,
	double v29,
	double v30,
	double v31,
	double v32,
	double v33,
	double v34,
	double v35,
	double v36,
	double v37,
	double v38,
	double v39,
	double v40,
	double v41,
	double v42,
	double v43,
	double v44,
	double v45,
	double v46,
	double v47,
	double v48,
	double v49,
	double v50,
	double v51,
	double v52,
	double v53,
	double v54,
	double v55,
	double v56,
	double v57,
	double v58,
	double v59,
	double v60,
	double v61,
	double v62,
	double v63,
	double v64,
	double v65,
	double v66,
	double v67,
	double v68,
	double v69
)
{
	double feature[] = {
		v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
		v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
		v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
		v31, v32, v33, v34, v35, v36, v37, v38, v39, v40,
		v41, v42, v43, v44, v45, v46, v47, v48, v49, v50,
		v51, v52, v53, v54, v55, v56, v57, v58, v59, v60,
		v61, v62, v63, v64, v65, v66, v67, v68, v69
	};
	int arrayNumber = sizeof feature / sizeof feature[0];
	wchar_t ws_endpoint[254];
	wchar_t ws_model_name[254];
	wchar_t ws_timestamp[254];
	wchar_t ws_expname[254];
	mbstowcs(ws_endpoint, endpoint, 254);
	mbstowcs(ws_model_name, model_name, 254);
	mbstowcs(ws_timestamp, timestamp, 254);
	mbstowcs(ws_expname, expname, 254);
	return _predict(ws_endpoint, ws_model_name, feature, arrayNumber, ws_timestamp, ws_expname, currentBar, NetProfit, OpenPositionProfit);
}

extern "C" double __declspec(dllexport) __stdcall backward(
	const char* endpoint,
	const char* model_name,
	double reward
)
{
		wchar_t ws_endpoint[254];
		wchar_t ws_model_name[254];
		mbstowcs(ws_endpoint, endpoint, 254);
		mbstowcs(ws_model_name, model_name, 254);
		return _backward(ws_endpoint, ws_model_name, reward);
}