from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from googleapiclient.http import MediaFileUpload
import numpy as np
import random
import time
import cv2

start_time = time.time()


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def delete_drive_service_file(service, file_id):
    service.files().delete(fileId=file_id).execute()


def update_file(service, update_drive_service_name, local_file_path, update_drive_service_folder_id):
    """
    將本地端的檔案傳到雲端上
    :param update_drive_service_folder_id: 判斷是否有 Folder id 沒有的話，會上到雲端的目錄
    :param service: 認證用
    :param update_drive_service_name: 存到 雲端上的名稱
    :param local_file_path: 本地端的位置
    :param local_file_name: 本地端的檔案名稱
    """
    print("正在上傳檔案...")
    if update_drive_service_folder_id is None:
        file_metadata = {'name': update_drive_service_name}
    else:
        print(update_drive_service_folder_id)
        file_metadata = {'name': update_drive_service_name,
                         'parents': update_drive_service_folder_id}

    media = MediaFileUpload(local_file_path, )
    file_metadata_size = media.size()
    start = time.time()
    file_id = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    end = time.time()
    print("上傳檔案成功！")
    print('雲端檔案名稱為: ' + str(file_metadata['name']))
    print('雲端檔案ID為: ' + str(file_id['id']))
    print('檔案大小為: ' + str(file_metadata_size) + ' byte')
    print("上傳時間為: " + str(end-start))

    return file_metadata['name'], file_id['id']


def search_file(service, update_drive_service_name, is_delete_search_file=False):
    """
    本地端
    取得到雲端名稱，可透過下載時，取得file id 下載
    :param service: 認證用
    :param update_drive_service_name: 要上傳到雲端的名稱
    :param is_delete_search_file: 判斷是否需要刪除這個檔案名稱
    :return:
    """
    # Call the Drive v3 API
    results = service.files().list(fields="nextPageToken, files(id, name)", spaces='drive',
                                   q="name = '" + update_drive_service_name + "' and trashed = false").execute()
    items = results.get('files', [])
    if not items:
        print('沒有發現你要找尋的 ' + update_drive_service_name + ' 檔案.')
    else:
        print('搜尋的檔案: ')
        for item in items:
            times = 1
            print(u'{0} ({1})'.format(item['name'], item['id']))
            if is_delete_search_file is True:
                print("刪除檔案為:" + u'{0} ({1})'.format(item['name'], item['id']))
                delete_drive_service_file(service, file_id=item['id'])

            if times == len(items):
                return item['id']
            else:
                times += 1

def search_folder(service, update_drive_folder_name=None):
    global response

    """
    如果雲端資料夾名稱相同，則只會選擇一個資料夾上傳，請勿取名相同名稱
    :param service: 認證用
    :param update_drive_folder_name: 取得指定資料夾的id，沒有的話回傳None，給錯也會回傳None
    :return:
    """
    get_folder_id_list = []
    print(len(get_folder_id_list))
    if update_drive_folder_name is not None:
        response = service.files().list(fields="nextPageToken, files(id, name)", spaces='drive',
                                        q = "name = '" + update_drive_folder_name +
                                        "' and mimeType = 'application/vnd.google-apps.folder' and trashed = false").execute()
        for file in response.get('files', []):
            # Process change
            # print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
            get_folder_id_list.append(file.get('id'))
        if len(get_folder_id_list) == 0:
            print("你給的資料夾名稱沒有在你的雲端上！，因此檔案會上傳至雲端根目錄")
            return None
        else:
            return get_folder_id_list
    return None

def main(is_update_file_function=False, update_drive_service_folder_name=None,
         update_drive_service_name=None, update_file_path=None,cred_file=None,token_file=None):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    print('*' * 10)

    if is_update_file_function is True:
        print(update_file_path + update_drive_service_name)
        print("=====執行上傳檔案=====")
        get_folder_id = search_folder(service = service, update_drive_folder_name = update_drive_service_folder_name)
        # 搜尋要上傳的檔案名稱是否有在雲端上並且刪除
        search_file(service=service, update_drive_service_name=update_drive_service_name,is_delete_search_file=True)
        time.sleep(0.2)
        # 檔案上傳到雲端上
        update_file(service=service, update_drive_service_name=update_drive_service_name,
                    local_file_path=os.getcwd() + '/' + update_drive_service_name, update_drive_service_folder_id=get_folder_id)
        print("=====上傳檔案完成=====")

#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party_result = np.zeros((6,img.shape[0]+ 2, img.shape[1]+ 2))
    
    img_2 = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)        # padding 
    
    for i in range (img_size[0] + 2):       
        for j in range(img_size[1] + 2):
            random_num = random.randint(-400,400)
            img_party_result[0][i][j] = (img_2[i][j] + random_num * 1) % 33292801
            img_party_result[1][i][j] = (img_2[i][j] + random_num * 2) % 33292801
            img_party_result[2][i][j] = (img_2[i][j] + random_num * 3) % 33292801
            img_party_result[3][i][j] = (img_2[i][j] + random_num * 4) % 33292801
            img_party_result[4][i][j] = (img_2[i][j] + random_num * 5) % 33292801
            img_party_result[5][i][j] = (img_2[i][j] + random_num * 6) % 33292801
            
            
    return img_party_result
        
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/test_01.bmp',0) 
    img_size = np.shape (img_original)

    print("Distributon")
    img_party = Distribution(img_original)   
    np.save('Party1', img_party[0])
    np.save('Party2', img_party[1])
    np.save('Party3', img_party[2]) 
    np.save('Party4', img_party[3]) 
    np.save('Party5', img_party[4]) 
    np.save('Party6', img_party[5]) 
      
    cred_token_party = ['peng01','peng03','peng05','chen03','chen16','chen21']
    
    for num in range (6):   
        print("--------Upload the file to Party{}--------".format(num+1))
        main(is_update_file_function=bool(True), update_drive_service_folder_name = 'New',
              update_drive_service_name='Party'+str(num+1)+'.npy', update_file_path=os.getcwd() + './',
              cred_file='credentials_'+cred_token_party[num]+'.json',token_file='token_'+cred_token_party[num]+'.pickle')


end_time = time.time()            
print("Time : ",round(end_time-start_time, 2),"sec")            