# MIUUL_Bootcamp

  Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.
  > Problem : Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine
 öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik
 mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

<sub>Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.</sub>

##### Pregnancies: Hamilelik sayısı
##### Glucose: Glikoz
##### BloodPressure: Kan basıncı (Diastolic(Küçük Tansiyon))
##### SkinThickness: Cilt Kalınlığı
##### Insulin: İnsülin.
##### BMI: Beden kitle indeksi.
##### DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
##### Age: Yaş (yıl)
##### Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

### GÖREV 1: KEŞİFCİ VERİ ANALİZİ
-  Adım 1: Genel resmi inceleyiniz.
-  Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
-  Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
-  Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
 - Adım 5: Aykırı gözlem analizi yapınız.
 - Adım 6: Eksik gözlem analizi yapınız.
- Adım 7: Korelasyon analizi yapınız.'''
    
###    GÖREV 2: FEATURE ENGINEERING
- Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta
ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili
  değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
   işlemleri uygulayabilirsiniz.
 - Adım 2: Yeni değişkenler oluşturunuz.
- Adım 3:  Encoding işlemlerini gerçekleştiriniz.
- Adım 4: Numerik değişkenler için standartlaştırma yapınız.
- Adım 5: Model oluşturunuz.


