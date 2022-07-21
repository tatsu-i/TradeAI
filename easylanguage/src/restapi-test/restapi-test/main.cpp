// stock-api.cpp : DLL 用にエクスポートされる関数を定義します。
//
#include <Windows.h>
#include <cpprest/http_client.h>
#include <locale.h>
#include <time.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace std;

double _predict(const wchar_t* ws_endpoint, const wchar_t* ws_symbolname, double* array, int arrayNumber)
{
	json::value postData;
	json::value feature = json::value::array();
	for (int i = 0; i < arrayNumber; i++) {
		feature[i] = array[i];
	}
	postData[L"model"] = json::value::string(ws_symbolname);
	postData[L"feature"] = feature;
	try
	{
		http_client client(ws_endpoint);
		auto task{ client.request(methods::POST, L"", postData.serialize(), L"application/json") };
		return task.then([ws_symbolname](http_response resp) {
			if (resp.status_code() == status_codes::OK)
			{
				return resp.extract_json();
			}
			}).then([ws_symbolname](json::value json) {
				// 応答結果を返す
				if (json[L"status"].as_integer() == 0) {
					return json[L"prediction"][0].as_double();
				}
				else {
					wofstream outputfile("C:\\stock-api-error.log", ios_base::out | ios_base::app);
					outputfile << "symbolname:" << ws_symbolname << " status:" << json[L"status"].as_integer() << " message:" << json[L"message"].as_string() << endl;
					outputfile.close();
				}
				return 0.0;
				}).get();
	}
	catch (const std::exception& e)
	{
		wofstream outputfile("C:\\stock-api-error.log", ios_base::out | ios_base::app);
		outputfile  << "symbolname:" << ws_symbolname << " Error:" << e.what() << endl;
		outputfile.close();
		return 0;
	}
}

int main(int argc, char* argv[])
{
	double array[] = { 1.47574819401444, 160.8588351431392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0 };
	int arrayNumber = sizeof array / sizeof array[0];
	if (argc != 4) {
		printf("restapi-test.exe [endpoint] [symbolname] [count]\n");
		return 1;
	}
	wchar_t ws_endpoint[254];
	wchar_t ws_symbolname[254];
	mbstowcs(ws_endpoint, argv[1], 254);
	mbstowcs(ws_symbolname, argv[2], 254);

	int count = atoi(argv[3]);
	clock_t start = clock();
	for (int i = 0; i < count; i++) {
		_predict(ws_endpoint, ws_symbolname, array, arrayNumber);
	}
	clock_t end = clock();
	const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
	printf("time %lf[ms]\n", time);
	return 0;
}