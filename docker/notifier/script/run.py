import httplib2, os
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
import slackweb
import time

# (1) Googleの開発者コンソールにログインしてGmail APIを有効にする
# https://console.developers.google.com/flows/enableapi?apiid=gmail

# (2) slackでIncoming Webhookの設定をする
# https://my.slack.com/services/new/incoming-webhook/

# Gmail権限のスコープを指定
SCOPES = 'https://www.googleapis.com/auth/gmail.modify'
# ダウンロードした権限ファイルのパス
CLIENT_SECRET_FILE = '/creds/client_id.json'
# ユーザーごとの設定ファイルの保存パス
USER_SECRET_FILE = '/creds/credentials-gmail.json'

# ユーザー認証データの取得
def gmail_user_auth():
    # ユーザーの認証データの読み取り
    store = Storage(USER_SECRET_FILE)
    credentials = store.get()
    # ユーザーが認証済みか?
    if not credentials or credentials.invalid:
        # 新規で認証する
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = 'Python Gmail API'
        credentials = tools.run_flow(flow, store, None)
        print('認証結果を保存しました:' + USER_SECRET_FILE)
    return credentials

# Gmailのサービスを取得
def gmail_get_service():
    # ユーザー認証の取得
    credentials = gmail_user_auth()
    http = credentials.authorize(httplib2.Http())
    # GmailのAPIを利用する
    service = discovery.build('gmail', 'v1', http=http)
    return service

def process_message(msg):
    return msg.replace('Order', '\nOrder').replace('Account:', '\nAccount:').replace('Occurred:', '\nOccurred:').replace('Signal:', '\nSignal:').replace('Workspace:', '\nWorkspace:').replace("Route:", "\nRoute:").replace("Duration:", "\nDuration:").replace("Qty Filled:", "\nQty Filled:")

# メッセージの一覧を取得
def gmail_get_messages():
    message_from = os.environ.get("MESSAGE_FROM")
    webhook_url = os.environ.get("WEBHOOK_URL")
    service = gmail_get_service()
    # メッセージの一覧を取得
    query = f'From: {message_from} Subject: TradeStation is:unread'
    messages = service.users().messages()
    msg_list = messages.list(userId='me',maxResults=100,q=query).execute()

    # 取得したメッセージの一覧を表示
    msg_len = len(msg_list.get('messages', []))
    print(f"fetch message {msg_len}")
    for msg in msg_list.get('messages', []):
        mail_id = msg['id']
        msg = messages.get(userId='me', id=mail_id).execute()
        snippet = msg['snippet']
        snippet = process_message(snippet)
        slack = slackweb.Slack(url=webhook_url)
        # 要約をslackに送信
        slack.notify(username="AI", icon_emoji=":yen", text=snippet) 
        #メールを既読にする
        messages.modify(userId="me", id=mail_id, body={"removeLabelIds": ["UNREAD"]}).execute()

if os.path.exists(CLIENT_SECRET_FILE):
    gmail_user_auth()
    print("メッセージの取得を実行")
    while True:
        try:
            gmail_get_messages()
        except Exception as e:
            print(e)
        time.sleep(10)
else:
    print(f"{CLIENT_SECRET_FILE} not found.")
