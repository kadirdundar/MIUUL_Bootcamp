# SALARY

 İş Problemi
 Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
 oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?



<sub>Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.</sub>

##### AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
##### Hits: 1986-1987 sezonundaki isabet sayısı
##### HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
##### Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
##### RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
##### Walks: Karşı oyuncuya yaptırılan hata sayısı
##### Years: Oyuncunun major liginde oynama süresi (sene)
##### CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
##### CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
##### CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
##### CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
##### CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
##### CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
##### League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
##### Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
##### PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
##### Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
##### Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
##### Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
##### NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



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


