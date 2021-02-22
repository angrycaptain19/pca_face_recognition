import os
import numpy as np
import cv2
import re
import shutil

if __name__  == "__main__":

    print("抓取 测试脸")
    # 抓取测试脸
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    test_images_root = ("test_images")
    test_faces_root = ("test_faces")
    eigen_faces_root = ("eigen_faces")
    files = os.listdir(test_images_root)


    for each in files:
        file_path = os.path.join(test_images_root, each)

        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 0)  # flag 大于0 三色，=0 灰色， <0 原本颜色

        faces = face_cascade.detectMultiScale(image, 1.1, 5)
        # 1.1 scale factor 表示人脸检测过程中每次迭代时图片的压缩率
        # 5  minNeighbors：每个人脸矩形保留近邻数目的最小值5

        for (x, y, w, h) in faces:
            X = int(x)
            W = min(int(x + w), image.shape[1])  # min 防止人脸横坐标超出图片
            Y = int(y)
            H = min(int(y + h), image.shape[0])  # min 防止人脸纵坐标超出图片

            image_face = cv2.resize(image[Y:H, X:W], (W - X, H - Y))  # 提取脸部图片
            size = (400, 400)
            image_face_resize = cv2.resize(image_face, size, interpolation=cv2.INTER_LANCZOS4)  # 脸部图片resize为400*400
            cv2.imencode('.jpg', image_face_resize)[1].tofile(os.path.join(test_faces_root, each))

            # 画人脸矩形

            #img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.namedWindow('Viking Detected!!')
            #cv2.imshow('Viking Detected!!', img)
            #cv2.waitKey(0)


    print("==================")
    print("抓取 明星脸")
    # 抓取明星脸
    star_images_root = ("star_images")
    star_faces_root = ("star_faces")
    files = os.listdir(star_images_root)

    for each in files:
        file_path = os.path.join(star_images_root, each)
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 0)  # flag 大于0 三色，=0 灰色， <0 原本颜色

        faces = face_cascade.detectMultiScale(image, 1.1, 5)
        # 1.1 scale factor 表示人脸检测过程中每次迭代时图片的压缩率
        # 5  minNeighbors：每个人脸矩形保留近邻数目的最小值5

        for (x, y, w, h) in faces:
            X = int(x)
            W = min(int(x + w), image.shape[1])  # min 防止人脸横坐标超出图片
            Y = int(y)
            H = min(int(y + h), image.shape[0])  # min 防止人脸纵坐标超出图片

            image_face = cv2.resize(image[Y:H, X:W], (W - X, H - Y))  # 提取脸部图片
            size = (400, 400)
            image_face_resize = cv2.resize(image_face, size, interpolation=cv2.INTER_LANCZOS4)  # 脸部图片resize为400*400
            cv2.imencode('.jpg', image_face_resize)[1].tofile(os.path.join(star_faces_root, each))


    print("==================")

    print("生成 训练集矩阵")

    star_images_root = ("star_images")
    star_faces_root = ("star_faces")
    files = os.listdir(star_faces_root)
    all = []
    stars = {}  # 明星脸字典
    for each in files:

        data = cv2.imdecode(np.fromfile(os.path.join(star_faces_root, each),dtype=np.uint8),0)
        vector = np.mat(data.reshape(data.size)).T
        all.append(vector)
        stars[each[0:-4]] = vector
    all = np.mat(np.array(all)).T   #160000 * 189

    average_value = np.mean(all,1)

    print("==================")
    #画平均脸
    print("生成 平均脸")
    average_face_data = average_value.reshape(400,400)
    average_face = cv2.imencode(".jpg",average_face_data)[1].tofile(os.path.join(eigen_faces_root, "明星平均脸.jpg"))
    print("==================")


    print("生成 投影矩阵")
    all = all - average_value  # 去中心化
    A = np.mat(all)  # 160000* 189
    B = A.T * A     # 189*189
    B_eig_value, B_eig_vector = np.linalg.eig(B)

    eig_val_index = np.argsort(B_eig_value)

    l = list(eig_val_index)
    l.reverse()
    eig_val_index = np.array(l)
    B_eig_vector = B_eig_vector[:, eig_val_index]

    C = B_eig_vector
    D = A*C        # 特征向量矩阵   # 160000 * 189
    # 坐标矩阵 = (D.T*D)逆*D.T = 单位矩阵逆*D.T = D.T
    # 投影矩阵 = D * (D.T*D)逆*D.T =D * 单位矩阵逆*D.T = D*D.T

    print("==================")

    # 画特征脸
    print("生成特征脸")
    b = D.shape[1]  # 得到b 特征脸的数量
    for index in range(b):
        eig_face_data = D[:, index]
        eig_face = eig_face_data.reshape(400, 400)
        cv2.imencode(".jpg", eig_face)[1].tofile(
            os.path.join(eigen_faces_root, ("特征脸{}.jpg".format(index+1))))

    # 投影矩阵归一化
    print("投影矩阵归一化")
    print("==================")
    a, b = D.shape
    mat = D
    for i in range(b):
        s = D[:, i]
        mc = np.power(np.sum(np.power(s, 2)), 1 / 2)
        mat[:, i] = D[:, i] / mc


    def get_zuobiao (face_data,matrix):
        zuobiao = matrix.T * face_data
        return zuobiao

    def calculate_distance (zuobiao1, zuobiao2):
        difference = zuobiao1- zuobiao2
        distance = np.power(np.sum(np.power(difference,2)),1/2)
        return distance


    print("坐标转换")

    # 把明星字典对应的数据换为投影后的坐标
    zuobiaos = {}
    for each_star in stars.keys():
        each_star_face_data = stars[each_star]
        each_star_face_zuobiao = get_zuobiao(each_star_face_data, D)
        zuobiaos[each_star] = each_star_face_zuobiao
    print("==================")
    print("完成 数据降维")
    print("==================")

    #坐标分类
    '''
    def zhengze (a):
        return re.findall('[\D]*', a)[0]
    
    zuobiaos2 = {}
    for each_star in zuobiaos.keys():
        current_zuobiao = zuobiaos[each_star]
        star = zhengze(each_star)
    
        if  star in zuobiaos2:
            zuobiaos2[star].append(current_zuobiao)
        else:
            zuobiaos2[star] = [current_zuobiao, ]
    
    print("坐标分类完成")
    '''
    # 正则表达式 去掉编号
    def zhengze (a):
        return re.findall('[\D]*', a)[0]

    print("开始识别 测试集")
    print("==================")

    differences = {}

    files = os.listdir(test_faces_root)
    for each_tested in files:
        test_data = cv2.imdecode(np.fromfile(os.path.join(test_faces_root, each_tested), dtype=np.uint8), 0)
        test_vector = np.mat(test_data.reshape(test_data.size)).T
        test_zuobiao = get_zuobiao(test_vector,D)

        for each_star in zuobiaos.keys():
            each_vector = zuobiaos[each_star]
            d = calculate_distance(each_vector, test_zuobiao)
            differences[each_star] = d

        minValue = min(differences.values())
        for i,j in differences.items():
            if j == minValue:
                star_face = cv2.imdecode(np.fromfile("star_faces/" + i + ".jpg", dtype=np.uint8), 0)  # flag 大于0 三色，=0 灰色， <0 原本颜色
                test_face = cv2.imdecode(np.fromfile("test_faces/" + each_tested, dtype=np.uint8), 0)  # flag 大于0 三色，=0 灰色， <0 原本颜色

                result = np.hstack((test_face, star_face))

                # 测试原图
                test_orig_root = "results/" + each_tested[0:-4] + " 像 " + zhengze(
                    i) + " a.jpg"

                shutil.copy("test_images/" + each_tested, test_orig_root)

                # 匹配的原图
                train_orig_root = "results/" + each_tested[0:-4] + " 像 " + zhengze(
                    i) + " b.jpg"
                shutil.copy("star_images/" + i + ".jpg", train_orig_root)

                # 对比图图
                cv2.imencode(".jpg", result)[1].tofile("results/" + each_tested[0:-4] + " 像 " + zhengze(i) + " c.jpg")

                # 原图投影
                new_root_c = "results/" + each_tested[0:-4] + " 像 " + zhengze(i) + " d.jpg"
                img_ty_orign = (mat * zuobiaos[i]).reshape(400,400)
                cv2.imencode('.jpg', img_ty_orign)[1].tofile(new_root_c)


                # 测试图投影
                new_root_d = "results/" + each_tested[
                                                                                         0:-4] + " 像 " + zhengze(
                    i) + " e.jpg"
                img_ty_test = (mat * test_zuobiao).reshape(400,400)
                cv2.imencode('.jpg', img_ty_test)[1].tofile(new_root_d)
                print("与    " + each_tested[0:-4] + "    最相似的是   " + zhengze(i))
