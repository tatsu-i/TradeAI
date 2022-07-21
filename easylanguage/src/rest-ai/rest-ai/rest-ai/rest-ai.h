// 以下の ifdef ブロックは、DLL からのエクスポートを容易にするマクロを作成するための
// 一般的な方法です。この DLL 内のすべてのファイルは、コマンド ラインで定義された RESTAI_EXPORTS
// シンボルを使用してコンパイルされます。このシンボルは、この DLL を使用するプロジェクトでは定義できません。
// ソースファイルがこのファイルを含んでいる他のプロジェクトは、
// RESTAI_API 関数を DLL からインポートされたと見なすのに対し、この DLL は、このマクロで定義された
// シンボルをエクスポートされたと見なします。
#ifdef RESTAI_EXPORTS
#define RESTAI_API __declspec(dllexport)
#else
#define RESTAI_API __declspec(dllimport)
#endif

// このクラスは dll からエクスポートされました
class RESTAI_API Crestai {
public:
	Crestai(void);
	// TODO: メソッドをここに追加します。
};

extern RESTAI_API int nrestai;

RESTAI_API int fnrestai(void);
