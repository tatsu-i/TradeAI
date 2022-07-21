#include "pch.h"
#include <cpprest/http_client.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace std;

int _slack_notify(const wchar_t* ws_webhook_url, const wchar_t* ws_text)
{
	json::value postData;
	postData[L"text"] = json::value::string(ws_text);
	postData[L"username"] = json::value::string(L"AI");
	postData[L"icon_emoji"] = json::value::string(L":yen");
	try
	{
		http_client client(ws_webhook_url);
		auto task{ client.request(methods::POST, L"", postData.serialize(), L"application/json") };
		return task.then([](http_response resp) {
			if (resp.status_code() == status_codes::OK)
			{
				return 0;
			}
			}).get();
	}
	catch (const std::exception & e)
	{
		wofstream outputfile("C:\\stock-ai.log", ios_base::out | ios_base::app);
		outputfile << "slack_notify error:" << e.what() << endl;
		outputfile.close();
		return 0;
	}
}

extern "C" int __declspec(dllexport) __stdcall slack_notify(
	const char* webhook_url,
	const char* text
)
{
	wchar_t ws_text[254];
	wchar_t ws_webhook_url[254];
	mbstowcs(ws_text, text, 254);
	mbstowcs(ws_webhook_url, webhook_url, 254);
	return _slack_notify(ws_webhook_url, ws_text);
}

int _mackerel_post_metric(const wchar_t* ws_service_url, const wchar_t* ws_api_key, double value)
{
	json::value postData;
	json::value postMetrics = json::value::array();
	postData[L"value"] = value;
	postData[L"time"] = time(NULL);
	postData[L"name"] = json::value::string(L"Strategy.Value");
	postMetrics[0] = postData;
	try
	{
		const method mtd = methods::POST;
		http_request msg(mtd);
		msg.headers().add(L"X-Api-Key", ws_api_key);
		msg.headers().add(L"Content-Type", "application/json");
		msg.set_body(postMetrics.serialize(), L"application/json");
		http_client client(ws_service_url);
		auto task{ client.request(msg) };
		return task.then([](http_response resp) {
			if (resp.status_code() == status_codes::OK)
			{
				return 0;
			}
			}).get();
	}
	catch (const std::exception & e)
	{
		wofstream outputfile("C:\\stock-ai.log", ios_base::out | ios_base::app);
		outputfile << "mackerel_post_metric error:" << e.what() << endl;
		outputfile.close();
		return 0;
	}
}

extern "C" int __declspec(dllexport) __stdcall mackerel_post_metric(
	const char* service_url,
	const char* api_key,
	double value
)
{
	wchar_t ws_service_url[254];
	wchar_t ws_api_key[254];
	wchar_t ws_hostId[254];
	mbstowcs(ws_service_url, service_url, 254);
	mbstowcs(ws_api_key, api_key, 254);
	return _mackerel_post_metric(ws_service_url, ws_api_key, value);
}