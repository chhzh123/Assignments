numBranches(X,L) :- setof(Branch,Year^Type^(restaurant(X,Year,Type),branch(X,Branch)),Z),length(Z,L).

%% ?- setof(Branch,Year^Type^(restaurant(Rest,Year,Type),branch(Rest,Branch)),Z),length(Z,X).
%% Rest = ajukejiacai,
%% Z = [shatainan, xintiandi, yongfu],
%% X = 3 ;
%% Rest = dagangxianmiaoshaoji,
%% Z = [beishan, cencun, changxing, dongpu, fangcun, gaosheng, huadong, kecun, nanpudadao|...],
%% X = 11 ;
%% Rest = diandude,
%% Z = [bainaohui, huachengdadao, huifudong, linhe, panfu, shiqiao, tianhebei, yangji, youtuobangshiguang|...],
%% X = 10 ;
%% Rest = hongmenyan,
%% Z = [xintiandi, zhilanwan],
%% X = 2 ;
%% Rest = huangmenjimifan,
%% Z = [beigang, dalingang, dongqu, dongxiaonan, pazhou, siyoubei, yuancun],
%% X = 7 ;
%% Rest = mixuebingcheng,
%% Z = [beigang, beiting, chentian, chisha, longdong, lujiang, shipaixi, shiqiao, wushan|...],
%% X = 12 ;
%% Rest = muwushaokao,
%% Z = [diwangguangchang, dongpu, gangding, heguang, runzhengguangchang, shayuan, shengdi, tangxia, tonghe|...],
%% X = 10 ;
%% Rest = shaxianxiaochi,
%% Z = [beigang, kangwangnan, luolang],
%% X = 3 ;
%% Rest = tongxianghui,
%% Z = [bainaohui, hanting, huizhoudasha, kaifadadao, maoshengdasha, shimaocheng, tianhebei, yongfu, yuanyangmingyuan|...],
%% X = 10 ;
%% Rest = yangguofu,
%% Z = [chebei, dayuan, shishangtianhe, xintiandi],
%% X = 4.

sameDistrict(X,Y) :- branch(X,Area1),branch(Y,Area2),district(Area1,Dist),district(Area2,Dist).