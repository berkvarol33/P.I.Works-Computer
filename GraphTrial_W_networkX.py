import numpy as np
import matplotlib.pyplot as plt



image1 = np.ones((5, 10))
image1[:, 0] = -1
image1[:, -1] = -1
image1[0, :] = -1
image1[-1, :] = -1

image2 = np.full((5, 10), -1)
image2[:, 0] = 1
image2[:, -1] = 1
image2[0, :] = 1
image2[-1, :] = 1


image_r = np.ones((5,5))
image_l = np.full((5, 5), -1)
image3 = np.concatenate((image_l,image_r), axis=1) 

image4 = np.concatenate((image_r,image_l), axis=1) 

def plot_images(image, x, y,title):
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.show()
    print(np.matrix.flatten(image))
     
    
def test_generator(image, x,y,title):
    plt.title(title)
    noisy = image + 0.2 * np.random.rand(x, y)
    noisy = noisy/noisy.max()
    plt.imshow(noisy, cmap='gray')
    plt.show()
    return noisy
def test_generator_rand(image, x,y,title):
    
    print(np.matrix.flatten(image))
    plt.title(title)
    rand_index = np.random.randint(0, x*y)
    rand_image = np.array(image)
    rand_image[int(rand_index % x),
    		int(rand_index / x)] = np.where(
    					rand_image[int(rand_index % x), int(rand_index / x)] == 1,0.0, 1.0)
    plt.imshow(rand_image, cmap='gray')
    plt.show()
    print(np.matrix.flatten(image))
    return  rand_image

print('Eğitim Kümesi ')
image1p = plot_images(image1, 5, 10,"image1")
image2p= plot_images(image2, 5, 10,"image2")
image3p = plot_images(image3, 5, 10,"image3")
image4p = plot_images(image4, 5, 10,"image4")

print('Test Kümesi')
test11 = test_generator(image1, 5, 10,"test1")
test12= test_generator_rand(image1,5,10,"test4 random")
test21 = test_generator(image2, 5, 10,"test2")
test22 =test_generator_rand(image2, 5,10,"test5 random")
test31 = test_generator(image3, 5, 10,"test3")
test32=test_generator_rand(image3, 5,10,"test6 random")
test41 = test_generator(image4, 5, 10,"test4")
test42=test_generator_rand(image4, 5,10,"test7 random")

print('DataSet')
X_egitim = np.vstack((image1.flatten(),image2.flatten(),image3.flatten(), image4.flatten()))

y_egitim = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]) 
y_test = np.array([1 , 1, 2, 2, 3, 3, 4, 4])


xe0=X_egitim[0]
xe1=X_egitim[1]
xe2=X_egitim[2]
xe3=X_egitim[3]

Xe0=test11.flatten() #X_test yazılıp ordan da çekilebilirdi ama hiç kullanmayacaktık
Xe1=test21.flatten()
Xe2=test31.flatten()
Xe3=test41.flatten()

Xe4=test12.flatten()
Xe5=test22.flatten()
Xe6=test32.flatten()
Xe7=test42.flatten()

#Ağırlık matrisinin oluşturulması

w0= np.dot(xe0[:,None],xe0[None,:])
w1= np.dot(xe1[:,None],xe1[None,:])
w2= np.dot(xe2[:,None],xe2[None,:])
w3= np.dot(xe3[:,None],xe3[None,:])


Wi= w0 + w1 + w2 + w3 
I= np.multiply(np.identity(50),2)
Wii= Wi - I
W= np.multiply(0.002,Wii)




# ayrık zaman hopfield ağı

def v(x):
 return np.dot(W,np.transpose(x))

def fi(v): #tüm elemanlar için ayrı ayrı w*x in sadece 1 ve -1 olabildiği hesaplamanın tanımlanması
    if v< 0:
     return -1
    if v >0:
      return 1 
    if v== 0:
      return v
  
def algoritma(x,y):
    V= v(x)
    f =[fi(i) for i in V]
    e= np.subtract(f,y) # çalıştırılan örünyüte göre xe0 xe1 xe2 xe3 olarak güncellenmeli!bu nedenle y var
    
    if e.any() != 0:
        F= np.reshape(f,(-1,10))
        plot_images(F, 5, 10,"doğru çizilmemiş plotlar")
        
        algoritma(f,y)
        
        
    if e.all() == 0:
        F= np.reshape(f,(-1,10))
        plot_images(F, 5, 10,"doğru çizilmiş plotlar")
        return print ("verilen değer için kararlı denge noktası")
   
#birinci_örüntü= algoritma(Xe0,xe0)
#ikinci_örüntü= algoritma(Xe1,xe1)
#üçüncü_örüntü =algoritma(Xe2,xe2)
#dördüncü_örüntü=algoritma(Xe3,xe3)
#beşinci_örüntü=algoritma(Xe4,xe0)
#altıncı_örüntü=algoritma(Xe5,xe1)
#yedinci_örüntü=algoritma(Xe6,xe2)
#sekizinci_örüntü=algoritma(Xe7,xe3)