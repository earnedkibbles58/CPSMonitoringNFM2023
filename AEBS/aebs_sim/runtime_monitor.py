

def prismViolationProbNothing(Pnn,Prn):
    if Pnn+Prn > 1:
        return -1

    violationProb = 209574*Prn**5*Pnn**25 + 5958238*Prn**6*Pnn**24 + 41883197*Prn**7*Pnn**23 - 54116781*Prn**8*Pnn**22 - 987705125*Prn**9*Pnn**21 - 2181119766*Prn**10*Pnn**20 - 7693830*Prn**11*Pnn**19 - 1301446398*Prn**12*Pnn**18 - 61204093111*Prn**13*Pnn**17 - 324987903141*Prn**14*Pnn**16 - 945824709608*Prn**15*Pnn**15 - 1714434407161*Prn**16*Pnn**14 - 1834018182958*Prn**17*Pnn**13 - 734062927913*Prn**18*Pnn**12 + 803948431551*Prn**19*Pnn**11 + 1386994525918*Prn**20*Pnn**10 + 863894227039*Prn**21*Pnn**9 + 204090880525*Prn**22*Pnn**8 - 40180895509*Prn**23*Pnn**7 - 35184233328*Prn**24*Pnn**6 - 6939058906*Prn**25*Pnn**5 - 60746334*Prn**26*Pnn**4 + 130434776*Prn**27*Pnn**3 + 11496141*Prn**28*Pnn**2 - 43585*Prn**29*Pnn - 17250*Prn**30 - 3555684*Prn**5*Pnn**24 - 70191440*Prn**6*Pnn**23 - 446720508*Prn**7*Pnn**22 + 512411893*Prn**8*Pnn**21 + 7943389526*Prn**9*Pnn**20 + 15948655824*Prn**10*Pnn**19 + 30402944320*Prn**11*Pnn**18 + 275064523205*Prn**12*Pnn**17 + 1561557601281*Prn**13*Pnn**16 + 5147662466064*Prn**14*Pnn**15 + 10566340629489*Prn**15*Pnn**14 + 12708695993429*Prn**16*Pnn**13 + 5534714286340*Prn**17*Pnn**12 - 7459224857695*Prn**18*Pnn**11 - 14056865025003*Prn**19*Pnn**10 - 9759921282694*Prn**20*Pnn**9 - 2624651703875*Prn**21*Pnn**8 + 494916943050*Prn**22*Pnn**7 + 514864658284*Prn**23*Pnn**6 + 114522409141*Prn**24*Pnn**5 + 2066116089*Prn**25*Pnn**4 - 2361920717*Prn**26*Pnn**3 - 235326176*Prn**27*Pnn**2 + 321816*Prn**28*Pnn + 386400*Prn**29 + 27487026*Prn**5*Pnn**23 + 325369353*Prn**6*Pnn**22 + 2048026456*Prn**7*Pnn**21 - 735447628*Prn**8*Pnn**20 - 22320115947*Prn**9*Pnn**19 - 76597830762*Prn**10*Pnn**18 - 532862487581*Prn**11*Pnn**17 - 3294652811551*Prn**12*Pnn**16 - 12493577938963*Prn**13*Pnn**15 - 29501942880849*Prn**14*Pnn**14 - 40710890594501*Prn**15*Pnn**13 - 20575251399378*Prn**16*Pnn**12 + 29886557880222*Prn**17*Pnn**11 + 64547146358997*Prn**18*Pnn**10 + 50858012755565*Prn**19*Pnn**9 + 15868331670337*Prn**20*Pnn**8 - 2672814705930*Prn**21*Pnn**7 - 3494063227022*Prn**22*Pnn**6 - 887540524937*Prn**23*Pnn**5 - 26595247773*Prn**24*Pnn**4 + 19990655853*Prn**25*Pnn**3 + 2278330208*Prn**26*Pnn**2 + 3679632*Prn**27*Pnn - 4098600*Prn**28 - 129471513*Prn**5*Pnn**22 
    violationProb += -486026017*Prn**6*Pnn**21 - 4506939603*Prn**7*Pnn**20 - 8452791842*Prn**8*Pnn**19 + 15738867785*Prn**9*Pnn**18 + 445417509358*Prn**10*Pnn**17 + 3851044934552*Prn**11*Pnn**16 + 17820505624897*Prn**12*Pnn**15 + 49697358148929*Prn**13*Pnn**14 + 80641447053246*Prn**14*Pnn**13 + 50840340767662*Prn**15*Pnn**12 - 67306826217650*Prn**16*Pnn**11 - 177745179534041*Prn**17*Pnn**10 - 162408390038851*Prn**18*Pnn**9 - 60076929798139*Prn**19*Pnn**8 + 7941094899854*Prn**20*Pnn**7 + 14548597127857*Prn**21*Pnn**6 + 4289291219362*Prn**22*Pnn**5 + 195135449808*Prn**23*Pnn**4 - 104572509429*Prn**24*Pnn**3 - 13854585943*Prn**25*Pnn**2 - 74155312*Prn**26*Pnn + 27311350*Prn**27 + 422195712*Prn**5*Pnn**21 - 2320038912*Prn**6*Pnn**20 - 1789056371*Prn**7*Pnn**19 + 40593777083*Prn**8*Pnn**18 - 11373429335*Prn**9*Pnn**17 - 2295247344725*Prn**10*Pnn**16 - 15779915795252*Prn**11*Pnn**15 - 55356154584198*Prn**12*Pnn**14 - 109414890232633*Prn**13*Pnn**13 - 90694725378901*Prn**14*Pnn**12 + 92267436586681*Prn**15*Pnn**11 + 327989213090085*Prn**16*Pnn**10 + 356491876478210*Prn**17*Pnn**9 + 159951760744871*Prn**18*Pnn**8 - 12089902529144*Prn**19*Pnn**7 - 41393859415270*Prn**20*Pnn**6 - 14476618847945*Prn**21*Pnn**5 - 955742346744*Prn**22*Pnn**4 + 375877234251*Prn**23*Pnn**3 + 59246357665*Prn**24*Pnn**2 + 600464634*Prn**25*Pnn - 127777650*Prn**26 - 1045650414*Prn**5*Pnn**20 + 17343569514*Prn**6*Pnn**19 + 56204601807*Prn**7*Pnn**18 - 15688126568*Prn**8*Pnn**17 + 509255237848*Prn**9*Pnn**16 + 8242909845425*Prn**10*Pnn**15 + 41052891988558*Prn**11*Pnn**14 + 104283071636206*Prn**12*Pnn**13 + 116811763715869*Prn**13*Pnn**12 - 76365399302294*Prn**14*Pnn**11 - 430255953485351*Prn**15*Pnn**10 - 572210560677896*Prn**16*Pnn**9 - 318277520968997*Prn**17*Pnn**8 - 1458037901928*Prn**18*Pnn**7 + 84664074722238*Prn**19*Pnn**6 + 36209675530588*Prn**20*Pnn**5 + 3388485865422*Prn**21*Pnn**4 - 972012676518*Prn**22*Pnn**3 - 188911314174*Prn**23*Pnn**2 - 3100454280*Prn**24*Pnn + 443646126*Prn**25 + 2177679200*Prn**5*Pnn**19 - 61663862420*Prn**6*Pnn**18 - 258175899807*Prn**7*Pnn**17 - 466784905622*Prn**8*Pnn**16 - 2948374036133*Prn**9*Pnn**15 - 20360189144770*Prn**10*Pnn**14 - 69757154329597*Prn**11*Pnn**13 
    violationProb += -107179946556719*Prn**12*Pnn**12 + 31429441704213*Prn**13*Pnn**11 + 415923360799095*Prn**14*Pnn**10 + 697020949481155*Prn**15*Pnn**9 + 489525122291207*Prn**16*Pnn**8 + 53832308871874*Prn**17*Pnn**7 - 127027492317558*Prn**18*Pnn**6 - 69503544359348*Prn**19*Pnn**5 - 9118886140428*Prn**20*Pnn**4 + 1821627417864*Prn**21*Pnn**3 + 464276747814*Prn**22*Pnn**2 + 11567987816*Prn**23*Pnn - 1174571475*Prn**24 - 4268819894*Prn**5*Pnn**18 + 152946274860*Prn**6*Pnn**17 + 767255505156*Prn**7*Pnn**16 + 2178268354843*Prn**8*Pnn**15 + 8935692188184*Prn**9*Pnn**14 + 35147453605028*Prn**10*Pnn**13 + 72320766595576*Prn**11*Pnn**12 + 7298063960213*Prn**12*Pnn**11 - 300937466093726*Prn**13*Pnn**10 - 657128079038825*Prn**14*Pnn**9 - 591205412186653*Prn**15*Pnn**8 - 142651400810542*Prn**16*Pnn**7 + 138864743011930*Prn**17*Pnn**6 + 104568202326496*Prn**18*Pnn**5 + 19185772841544*Prn**19*Pnn**4 - 2357822277156*Prn**20*Pnn**3 - 895234199607*Prn**21*Pnn**2 - 33057085248*Prn**22*Pnn + 2383822650*Prn**23 + 8372420949*Prn**5*Pnn**17 - 292069533195*Prn**6*Pnn**16 - 1694474779536*Prn**7*Pnn**15 - 5621530928444*Prn**8*Pnn**14 - 17742864726179*Prn**9*Pnn**13 - 41802618241341*Prn**10*Pnn**12 - 26232010163630*Prn**11*Pnn**11 + 157662868946044*Prn**12*Pnn**10 + 479234067019842*Prn**13*Pnn**9 + 561297709051563*Prn**14*Pnn**8 + 225107393069130*Prn**15*Pnn**7 - 104841531112942*Prn**16*Pnn**6 - 124793818590896*Prn**17*Pnn**5 - 32145393513804*Prn**18*Pnn**4 + 1622658667212*Prn**19*Pnn**3 + 1361798082282*Prn**20*Pnn**2 + 74755478853*Prn**21*Pnn - 3617573400*Prn**22 - 15887912964*Prn**5*Pnn**16 + 445514336383*Prn**6*Pnn**15 + 2889689504113*Prn**7*Pnn**14 + 9940835409045*Prn**8*Pnn**13 + 24744481694141*Prn**9*Pnn**12 + 30847777933650*Prn**10*Pnn**11 - 47718749538069*Prn**11*Pnn**10 - 261615749157310*Prn**12*Pnn**9 - 413500201607977*Prn**13*Pnn**8 - 247888358526952*Prn**14*Pnn**7 + 43244861799854*Prn**15*Pnn**6 + 118633357925550*Prn**16*Pnn**5 + 43340728349022*Prn**17*Pnn**4 + 948855334322*Prn**18*Pnn**3 - 1619846950920*Prn**19*Pnn**2 - 136423659856*Prn**20*Pnn + 3689612850*Prn**21 + 27306631880*Prn**5*Pnn**15 - 546886173652*Prn**6*Pnn**14 - 3845213550666*Prn**7*Pnn**13 - 12790339884241*Prn**8*Pnn**12
    violationProb += -24373371846147*Prn**9*Pnn**11 - 6265233731057*Prn**10*Pnn**10 + 96426834548035*Prn**11*Pnn**9 + 229383179028839*Prn**12*Pnn**8 + 199107180490218*Prn**13*Pnn**7 + 9293243975158*Prn**14*Pnn**6 - 89380444213918*Prn**15*Pnn**5 - 47195947485102*Prn**16*Pnn**4 - 4649439318602*Prn**17*Pnn**3 + 1455205023976*Prn**18*Pnn**2 + 203186121600*Prn**19*Pnn - 1186559880*Prn**20 - 40834340199*Prn**5*Pnn**14 + 531102872550*Prn**6*Pnn**13 + 3967678154624*Prn**7*Pnn**12 + 12060760747897*Prn**8*Pnn**11 + 15824661502663*Prn**9*Pnn**10 - 15756404425493*Prn**10*Pnn**9 - 90543889667389*Prn**11*Pnn**8 - 117824879041038*Prn**12*Pnn**7 - 31122555370036*Prn**13*Pnn**6 + 52218582070148*Prn**14*Pnn**5 + 41366356625880*Prn**15*Pnn**4 + 7648643058810*Prn**16*Pnn**3 - 882181825486*Prn**17*Pnn**2 - 248181613344*Prn**18*Pnn - 4278752100*Prn**19 + 52425047030*Prn**5*Pnn**13 - 385122876323*Prn**6*Pnn**12 - 3080806944190*Prn**7*Pnn**11 - 7969679923191*Prn**8*Pnn**10 - 4576065010790*Prn**9*Pnn**9 + 22732152933625*Prn**10*Pnn**8 + 51253031254984*Prn**11*Pnn**7 + 27827513306606*Prn**12*Pnn**6 - 22111983390406*Prn**13*Pnn**5 - 28803735249744*Prn**14*Pnn**4 - 8363812791750*Prn**15*Pnn**3 + 175181028698*Prn**16*Pnn**2 + 248287491452*Prn**17*Pnn + 11092239900*Prn**18 - 57677949396*Prn**5*Pnn**12 + 169275781255*Prn**6*Pnn**11 + 1638377310127*Prn**7*Pnn**10 + 3015793471084*Prn**8*Pnn**9 - 2836960105499*Prn**9*Pnn**8 - 16619002702912*Prn**10*Pnn**7 - 16391084020262*Prn**11*Pnn**6 + 5077783037756*Prn**12*Pnn**5 + 15457287742584*Prn**13*Pnn**4 + 6732276679488*Prn**14*Pnn**3 + 332814498412*Prn**15*Pnn**2 - 201621039312*Prn**16*Pnn - 16222182900*Prn**17 + 54376305924*Prn**5*Pnn**11 + 20339023389*Prn**6*Pnn**10 - 367810385875*Prn**7*Pnn**9 + 286584567281*Prn**8*Pnn**8 + 4486656432846*Prn**9*Pnn**7 + 7506529828302*Prn**10*Pnn**6 + 1216776176060*Prn**11*Pnn**5 - 5937705590412*Prn**12*Pnn**4 - 4065105764772*Prn**13*Pnn**3 - 492855318780*Prn**14*Pnn**2 + 130019505264*Prn**15*Pnn + 17265181725*Prn**16 - 43834225880*Prn**5*Pnn**10 - 118507025503*Prn**6*Pnn**9 - 291318637067*Prn**7*Pnn**8 - 1294929013906*Prn**8*Pnn**7 - 2914467559818*Prn**9*Pnn**6 - 1974796814704*Prn**10*Pnn**5 + 1251979682424*Prn**11*Pnn**4
    violationProb += 1797211195404*Prn**12*Pnn**3 + 392847400506*Prn**13*Pnn**2 - 63257043520*Prn**14*Pnn - 14174977740*Prn**15 + 30059062307*Prn**5*Pnn**9 + 125338548232*Prn**6*Pnn**8 + 393550167739*Prn**7*Pnn**7 + 965850052310*Prn**8*Pnn**6 + 1137034650434*Prn**9*Pnn**5 + 170238301818*Prn**10*Pnn**4 - 528282206844*Prn**11*Pnn**3 - 215588390391*Prn**12*Pnn**2 + 19880516601*Prn**13*Pnn + 9058551150*Prn**14 - 17385040884*Prn**5*Pnn**8 - 84997152762*Prn**6*Pnn**7 - 248409166514*Prn**7*Pnn**6 - 416764641403*Prn**8*Pnn**5 - 261667546527*Prn**9*Pnn**4 + 63425876931*Prn**10*Pnn**3 + 83967202968*Prn**11*Pnn**2 - 777695688*Prn**12*Pnn - 4399847100*Prn**13 + 8371150928*Prn**5*Pnn**7 + 41658177472*Prn**6*Pnn**6 + 101148847903*Prn**7*Pnn**5 + 111163730787*Prn**8*Pnn**4 + 24928600229*Prn**9*Pnn**3 - 21959310648*Prn**10*Pnn**2 - 3495727312*Prn**11*Pnn + 1494015600*Prn**12 - 3290237261*Prn**5*Pnn**6 - 14946560790*Prn**6*Pnn**5 - 27018320616*Prn**7*Pnn**4 - 15861256301*Prn**8*Pnn**3 + 2974993285*Prn**9*Pnn**2 + 2449335504*Prn**10*Pnn - 243697650*Prn**11 + 1022809815*Prn**5*Pnn**5 + 3775374840*Prn**6*Pnn**4 + 4203519315*Prn**7*Pnn**3 + 219487301*Prn**8*Pnn**2 - 985189590*Prn**9*Pnn - 73160010*Prn**10 - 237892734*Prn**5*Pnn**4 - 594684738*Prn**6*Pnn**3 - 185433886*Prn**7*Pnn**2 + 265627736*Prn**8*Pnn + 69714150*Prn**9 + 36732372*Prn**5*Pnn**3 + 34428358*Prn**6*Pnn**2 - 47802648*Prn**7*Pnn - 26601225*Prn**8 - 2421603*Prn**5*Pnn**2 + 5246208*Prn**6*Pnn + 6116850*Prn**7 - 267421*Prn**5*Pnn - 834900*Prn**6 + 53130*Prn**5
    return violationProb



def prismViolationProbRock(Pnr,Prr):
    if Pnr+Prr > 1:
        return -1

    violationProb = 23800*Prr*Pnr**28 - 107277*Prr**2*Pnr**27 - 2495655*Prr**3*Pnr**26 + 35157962*Prr**4*Pnr**25 + 592816655*Prr**5*Pnr**24 + 3498146967*Prr**6*Pnr**23 + 11004769704*Prr**7*Pnr**22 + 18512518880*Prr**8*Pnr**21 + 8359871993*Prr**9*Pnr**20 - 30950993665*Prr**10*Pnr**19 - 72645254409*Prr**11*Pnr**18 - 74629462150*Prr**12*Pnr**17 - 42586153886*Prr**13*Pnr**16 - 18694189718*Prr**14*Pnr**15 - 15063934167*Prr**15*Pnr**14 - 13513804280*Prr**16*Pnr**13 - 7156330780*Prr**17*Pnr**12 - 1991054490*Prr**18*Pnr**11 - 227860302*Prr**19*Pnr**10 + 15500959*Prr**20*Pnr**9 + 5155428*Prr**21*Pnr**8 - 2194107*Prr**22*Pnr**7 - 967840*Prr**23*Pnr**6 - 5194*Prr**24*Pnr**5 + 40975*Prr**25*Pnr**4 - 532*Prr**26*Pnr**3 - 998*Prr**27*Pnr**2 - 495040*Prr*Pnr**27 + 942520*Prr**2*Pnr**26 + 28534890*Prr**3*Pnr**25 - 604834230*Prr**4*Pnr**24 - 8291440953*Prr**5*Pnr**23 - 42236737603*Prr**6*Pnr**22 - 111134607309*Prr**7*Pnr**21 - 139184621743*Prr**8*Pnr**20 + 17727959106*Prr**9*Pnr**19 + 323939383371*Prr**10*Pnr**18 + 467513700743*Prr**11*Pnr**17 + 300629134894*Prr**12*Pnr**16 + 96105598245*Prr**13*Pnr**15 + 70271883928*Prr**14*Pnr**14 + 102084249718*Prr**15*Pnr**13 + 71304376091*Prr**16*Pnr**12 + 22530104550*Prr**17*Pnr**11 + 2061874447*Prr**18*Pnr**10 - 470524277*Prr**19*Pnr**9 - 60292212*Prr**20*Pnr**8 + 44559520*Prr**21*Pnr**7 + 16889206*Prr**22*Pnr**6 + 574153*Prr**23*Pnr**5 - 698167*Prr**24*Pnr**4 - 9031*Prr**25*Pnr**3 + 20030*Prr**26*Pnr**2 + 392*Prr**27*Pnr + 4876200*Prr*Pnr**26 + 837632*Prr**2*Pnr**25 - 125838528*Prr**3*Pnr**24 + 4627478594*Prr**4*Pnr**23 + 51999607324*Prr**5*Pnr**22 + 224765750609*Prr**6*Pnr**21 + 477060166738*Prr**7*Pnr**20 + 381949902307*Prr**8*Pnr**19 - 411168293491*Prr**9*Pnr**18 - 1218697228266*Prr**10*Pnr**17 - 1019391457341*Prr**11*Pnr**16 - 249428570372*Prr**12*Pnr**15 - 42006270519*Prr**13*Pnr**14 - 285743694301*Prr**14*Pnr**13 - 299691692585*Prr**15*Pnr**12 - 111593988595*Prr**16*Pnr**11 - 6459926143*Prr**17*Pnr**10 + 5034343592*Prr**18*Pnr**9 + 365967237*Prr**19*Pnr**8 - 444694942*Prr**20*Pnr**7 - 147349969*Prr**21*Pnr**6 - 8186333*Prr**22*Pnr**5 + 5634179*Prr**23*Pnr**4 + 243641*Prr**24*Pnr**3 - 188403*Prr**25*Pnr**2 - 7693*Prr**26*Pnr
    violationProb += -30220512*Prr*Pnr**25 - 52980854*Prr**2*Pnr**24 + 190639631*Prr**3*Pnr**23 - 20712103620*Prr**4*Pnr**22 - 190843248834*Prr**5*Pnr**21 - 681269179409*Prr**6*Pnr**20 - 1087565008448*Prr**7*Pnr**19 - 275089962414*Prr**8*Pnr**18 + 1592699473956*Prr**9*Pnr**17 + 2121171770313*Prr**10*Pnr**16 + 627375357257*Prr**11*Pnr**15 - 294035161909*Prr**12*Pnr**14 + 320579295661*Prr**13*Pnr**13 + 709744282469*Prr**14*Pnr**12 + 323320683559*Prr**15*Pnr**11 + 14868980*Prr**16*Pnr**10 - 29626627952*Prr**17*Pnr**9 - 1596485123*Prr**18*Pnr**8 + 2810708691*Prr**19*Pnr**7 + 855176684*Prr**20*Pnr**6 + 58220688*Prr**21*Pnr**5 - 28996902*Prr**22*Pnr**4 - 2268970*Prr**23*Pnr**3 + 1103487*Prr**24*Pnr**2 + 71574*Prr**25*Pnr + 132031890*Prr*Pnr**24 + 379265371*Prr**2*Pnr**23 + 472769977*Prr**3*Pnr**22 + 59276402079*Prr**4*Pnr**21 + 443679104031*Prr**5*Pnr**20 + 1241273104835*Prr**6*Pnr**19 + 1218598514761*Prr**7*Pnr**18 - 950546516645*Prr**8*Pnr**17 - 2937906468615*Prr**9*Pnr**16 - 1560101008462*Prr**10*Pnr**15 + 597731255441*Prr**11*Pnr**14 + 42832417988*Prr**12*Pnr**13 - 1077633323955*Prr**13*Pnr**12 - 622798591980*Prr**14*Pnr**11 + 63984732367*Prr**15*Pnr**10 + 113133424934*Prr**16*Pnr**9 + 5509423630*Prr**17*Pnr**8 - 12443326150*Prr**18*Pnr**7 - 3674297344*Prr**19*Pnr**6 - 265497944*Prr**20*Pnr**5 + 108982560*Prr**21*Pnr**4 + 12665716*Prr**22*Pnr**3 - 4507298*Prr**23*Pnr**2 - 419674*Prr**24*Pnr - 431681904*Prr*Pnr**23 - 1537106103*Prr**2*Pnr**22 - 2706941223*Prr**3*Pnr**21 - 108171906026*Prr**4*Pnr**20 - 637805572516*Prr**5*Pnr**19 - 1195840202182*Prr**6*Pnr**18 + 116052780766*Prr**7*Pnr**17 + 2978817417754*Prr**8*Pnr**16 + 2979007463222*Prr**9*Pnr**15 + 68977045841*Prr**10*Pnr**14 - 378183115618*Prr**11*Pnr**13 + 1190059214371*Prr**12*Pnr**12 + 878962680491*Prr**13*Pnr**11 - 237540677297*Prr**14*Pnr**10 - 304388634574*Prr**15*Pnr**9 - 15042834268*Prr**16*Pnr**8 + 40827568311*Prr**17*Pnr**7 + 12271662296*Prr**18*Pnr**6 + 865982344*Prr**19*Pnr**5 - 327247387*Prr**20*Pnr**4 - 49703892*Prr**21*Pnr**3 + 13613926*Prr**22*Pnr**2 + 1738909*Prr**23*Pnr + 56*Prr**24 + 1093986576*Prr*Pnr**22 + 4207626946*Prr**2*Pnr**21 + 4219785439*Prr**3*Pnr**20 + 104034363427*Prr**4*Pnr**19
    violationProb += 414107944643*Prr**5*Pnr**18 - 92503011658*Prr**6*Pnr**17 - 2597349900007*Prr**7*Pnr**16 - 4166637740593*Prr**8*Pnr**15 - 1906176489152*Prr**9*Pnr**14 - 125560434911*Prr**10*Pnr**13 - 1194265368207*Prr**11*Pnr**12 - 1016136034693*Prr**12*Pnr**11 + 483992101381*Prr**13*Pnr**10 + 604288199015*Prr**14*Pnr**9 + 31863825697*Prr**15*Pnr**8 - 102955607432*Prr**16*Pnr**7 - 32655212183*Prr**17*Pnr**6 - 2144869068*Prr**18*Pnr**5 + 838882973*Prr**19*Pnr**4 + 148798820*Prr**20*Pnr**3 - 31416339*Prr**21*Pnr**2 - 5410711*Prr**22*Pnr - 979*Prr**23 - 2195066712*Prr*Pnr**21 - 8238258065*Prr**2*Pnr**20 + 4122780281*Prr**3*Pnr**19 + 38203780438*Prr**4*Pnr**18 + 395821966438*Prr**5*Pnr**17 + 2117478881051*Prr**6*Pnr**16 + 4370647850172*Prr**7*Pnr**15 + 3735406771558*Prr**8*Pnr**14 + 1472850396347*Prr**9*Pnr**13 + 1367209495181*Prr**10*Pnr**12 + 1081274274732*Prr**11*Pnr**11 - 642294757836*Prr**12*Pnr**10 - 908889490629*Prr**13*Pnr**9 - 51754191253*Prr**14*Pnr**8 + 204459919895*Prr**15*Pnr**7 + 70220061130*Prr**16*Pnr**6 + 4178690718*Prr**17*Pnr**5 - 1909182720*Prr**18*Pnr**4 - 358411593*Prr**19*Pnr**3 + 56298778*Prr**20*Pnr**2 + 13113784*Prr**21*Pnr + 8055*Prr**22 + 3529047708*Prr*Pnr**20 + 11641304278*Prr**2*Pnr**19 - 33430503682*Prr**3*Pnr**18 - 319077014252*Prr**4*Pnr**17 - 1394501568279*Prr**5*Pnr**16 - 3415984351541*Prr**6*Pnr**15 - 4206857105154*Prr**7*Pnr**14 - 2666419220090*Prr**8*Pnr**13 - 1658828319980*Prr**9*Pnr**12 - 1108326636155*Prr**10*Pnr**11 + 567669199842*Prr**11*Pnr**10 + 1049720929472*Prr**12*Pnr**9 + 64140717258*Prr**13*Pnr**8 - 324929054185*Prr**14*Pnr**7 - 123146385185*Prr**15*Pnr**6 - 6530302855*Prr**16*Pnr**5 + 3889100324*Prr**17*Pnr**4 + 722223355*Prr**18*Pnr**3 - 78461015*Prr**19*Pnr**2 - 25340802*Prr**20*Pnr - 41426*Prr**21 - 4566443488*Prr*Pnr**19 - 11353935654*Prr**2*Pnr**18 + 81276606870*Prr**3*Pnr**17 + 593420304264*Prr**4*Pnr**16 + 1893548527868*Prr**5*Pnr**15 + 3173493906826*Prr**6*Pnr**14 + 2815043047034*Prr**7*Pnr**13 + 1749916921733*Prr**8*Pnr**12 + 1044206557641*Prr**9*Pnr**11 - 301587645737*Prr**10*Pnr**10 - 933807207347*Prr**11*Pnr**9 - 60411476698*Prr**12*Pnr**8 + 417206414208*Prr**13*Pnr**7 + 177236947616*Prr**14*Pnr**6
    violationProb += 8246667149*Prr**15*Pnr**5 - 7000841136*Prr**16*Pnr**4 - 1252930895*Prr**17*Pnr**3 + 83324859*Prr**18*Pnr**2 + 39633247*Prr**19*Pnr + 149149*Prr**20 + 4740999740*Prr*Pnr**18 + 6117300827*Prr**2*Pnr**17 - 123771051573*Prr**3*Pnr**16 - 680908195601*Prr**4*Pnr**15 - 1630171961389*Prr**5*Pnr**14 - 1998938230453*Prr**6*Pnr**13 - 1454821696512*Prr**7*Pnr**12 - 849518157399*Prr**8*Pnr**11 + 31199861572*Prr**9*Pnr**10 + 634887955316*Prr**10*Pnr**9 + 43049639638*Prr**11*Pnr**8 - 434603168270*Prr**12*Pnr**7 - 210157591968*Prr**13*Pnr**6 - 8380322922*Prr**14*Pnr**5 + 10942894522*Prr**15*Pnr**4 + 1906533244*Prr**16*Pnr**3 - 62405431*Prr**17*Pnr**2 - 50642251*Prr**18*Pnr - 398944*Prr**19 - 3898260988*Prr*Pnr**17 + 1501238577*Prr**2*Pnr**16 + 133153672441*Prr**3*Pnr**15 + 545243478411*Prr**4*Pnr**14 + 973727425610*Prr**5*Pnr**13 + 912823387725*Prr**6*Pnr**12 + 573036005363*Prr**7*Pnr**11 + 104923899735*Prr**8*Pnr**10 - 322383821665*Prr**9*Pnr**9 - 23362059576*Prr**10*Pnr**8 + 366968905238*Prr**11*Pnr**7 + 205609165184*Prr**12*Pnr**6 + 6729970902*Prr**13*Pnr**5 - 14658732902*Prr**14*Pnr**4 - 2567004614*Prr**15*Pnr**3 + 21894467*Prr**16*Pnr**2 + 53150225*Prr**17*Pnr + 820911*Prr**18 + 2461634836*Prr*Pnr**16 - 6981504506*Prr**2*Pnr**15 - 104995198611*Prr**3*Pnr**14 - 314333673701*Prr**4*Pnr**13 - 416132762219*Prr**5*Pnr**12 - 309783184217*Prr**6*Pnr**11 - 111253377457*Prr**7*Pnr**10 + 115800558683*Prr**8*Pnr**9 + 10338875299*Prr**9*Pnr**8 - 249649525752*Prr**10*Pnr**7 - 165789698774*Prr**11*Pnr**6 - 4095142786*Prr**12*Pnr**5 + 16701227505*Prr**13*Pnr**4 + 3060799795*Prr**14*Pnr**3 + 21147395*Prr**15*Pnr**2 - 45958109*Prr**16*Pnr - 1327326*Prr**17 - 1106878618*Prr*Pnr**15 + 7913429524*Prr**2*Pnr**14 + 61089933630*Prr**3*Pnr**13 + 130796787591*Prr**4*Pnr**12 + 128660183569*Prr**5*Pnr**11 + 65907338288*Prr**6*Pnr**10 - 25142768537*Prr**7*Pnr**9 - 4661168272*Prr**8*Pnr**8 + 135192492204*Prr**9*Pnr**7 + 109724217203*Prr**10*Pnr**6 + 1696292779*Prr**11*Pnr**5 - 16116535139*Prr**12*Pnr**4 - 3219722361*Prr**13*Pnr**3 - 51755425*Prr**14*Pnr**2 + 32880377*Prr**15*Pnr + 1707277*Prr**16 + 268723334*Prr*Pnr**14 - 5562934187*Prr**2*Pnr**13 - 25933055823*Prr**3*Pnr**12
    violationProb += -38788387503*Prr**4*Pnr**11 - 26745566991*Prr**5*Pnr**10 + 739036739*Prr**6*Pnr**9 + 2560855641*Prr**7*Pnr**8 - 57116920951*Prr**8*Pnr**7 - 59149276577*Prr**9*Pnr**6 - 290491823*Prr**10*Pnr**5 + 13131903125*Prr**11*Pnr**4 + 2971554000*Prr**12*Pnr**3 + 65698259*Prr**13*Pnr**2 - 19765427*Prr**14*Pnr - 1756898*Prr**15 + 48389494*Prr*Pnr**13 + 2688879608*Prr**2*Pnr**12 + 7802391874*Prr**3*Pnr**11 + 7722266438*Prr**4*Pnr**10 + 1600355271*Prr**5*Pnr**9 - 1368446756*Prr**6*Pnr**8 + 18226931386*Prr**7*Pnr**7 + 25660039254*Prr**8*Pnr**6 - 168443364*Prr**9*Pnr**5 - 9001963063*Prr**10*Pnr**4 - 2392321742*Prr**11*Pnr**3 - 66959119*Prr**12*Pnr**2 + 10524459*Prr**13*Pnr + 1446446*Prr**14 - 80805442*Prr*Pnr**12 - 896449431*Prr**2*Pnr**11 - 1550078463*Prr**3*Pnr**10 - 635643391*Prr**4*Pnr**9 + 540873651*Prr**5*Pnr**8 - 4162134285*Prr**6*Pnr**7 - 8799663989*Prr**7*Pnr**6 + 155805831*Prr**8*Pnr**5 + 5164256553*Prr**9*Pnr**4 + 1669897247*Prr**10*Pnr**3 + 60536403*Prr**11*Pnr**2 - 5646689*Prr**12*Pnr - 947324*Prr**13 + 39604094*Prr*Pnr**11 + 194895530*Prr**2*Pnr**10 + 145214396*Prr**3*Pnr**9 - 135073308*Prr**4*Pnr**8 + 614960319*Prr**5*Pnr**7 + 2324134796*Prr**6*Pnr**6 - 60422042*Prr**7*Pnr**5 - 2460333551*Prr**8*Pnr**4 - 1003305220*Prr**9*Pnr**3 - 49302497*Prr**10*Pnr**2 + 3539193*Prr**11*Pnr + 487578*Prr**12 - 11148354*Prr*Pnr**10 - 22642891*Prr**2*Pnr**9 + 16844937*Prr**3*Pnr**8 - 46301905*Prr**4*Pnr**7 - 454851417*Prr**5*Pnr**6 + 13359473*Prr**6*Pnr**5 + 962909249*Prr**7*Pnr**4 + 513797967*Prr**8*Pnr**3 + 35465312*Prr**9*Pnr**2 - 2521444*Prr**10*Pnr - 193340*Prr**11 + 1842886*Prr*Pnr**9 + 30042*Prr**2*Pnr**8 + 497058*Prr**3*Pnr**7 + 62093740*Prr**4*Pnr**6 - 2789482*Prr**5*Pnr**5 - 305007702*Prr**6*Pnr**4 - 221188524*Prr**7*Pnr**3 - 21923038*Prr**8*Pnr**2 + 1751159*Prr**9*Pnr + 57750*Prr**10 - 180992*Prr*Pnr**8 - 55445*Prr**2*Pnr**7 - 5309332*Prr**3*Pnr**6 + 1402436*Prr**4*Pnr**5 + 76607853*Prr**5*Pnr**4 + 78487185*Prr**6*Pnr**3 + 11357707*Prr**7*Pnr**2 - 1070701*Prr**8*Pnr - 13684*Prr**9 + 26594*Prr*Pnr**7 + 216667*Prr**2*Pnr**6 - 620543*Prr**3*Pnr**5 - 14815932*Prr**4*Pnr**4 - 22316547*Prr**5*Pnr**3 - 4816361*Prr**6*Pnr**2 + 552662*Prr**7*Pnr + 4506*Prr**8
    violationProb += 802*Prr*Pnr**6 + 140593*Prr**2*Pnr**5 + 2105312*Prr**3*Pnr**4 + 4875752*Prr**4*Pnr**3 + 1626852*Prr**5*Pnr**2 - 235330*Prr**6*Pnr - 3523*Prr**7 - 12710*Prr*Pnr**5 - 200407*Prr**2*Pnr**4 - 765741*Prr**3*Pnr**3 - 421359*Prr**4*Pnr**2 + 80545*Prr**5*Pnr + 3003*Prr**6 + 9878*Prr*Pnr**4 + 76578*Prr**2*Pnr**3 + 78718*Prr**3*Pnr**2 - 21305*Prr**4*Pnr - 2002*Prr**5 - 3638*Prr*Pnr**3 - 9460*Prr**2*Pnr**2 + 4084*Prr**3*Pnr + 1001*Prr**4 + 550*Prr*Pnr**2 - 504*Prr**2*Pnr - 364*Prr**3 + 30*Prr*Pnr + 91*Prr**2 - 14*Prr + 1
    return violationProb

def prismViolationProbCar(Pnc,Prc):
    if Pnc+Prc > 1:
        return -1

    violationProb = (-38032)*Prc**9*Pnc**21 - 1623534*Prc**10*Pnc**20 - 30454682*Prc**11*Pnc**19 - 302423018*Prc**12*Pnc**18 - 1730017212*Prc**13*Pnc**17 - 5748603591*Prc**14*Pnc**16 - 9613185250*Prc**15*Pnc**15 + 786598284*Prc**16*Pnc**14 + 40137060891*Prc**17*Pnc**13 + 92177211764*Prc**18*Pnc**12 + 113228858516*Prc**19*Pnc**11 + 87055282969*Prc**20*Pnc**10 + 43625652463*Prc**21*Pnc**9 + 14290326056*Prc**22*Pnc**8 + 2992151368*Prc**23*Pnc**7 + 379797612*Prc**24*Pnc**6 + 26157186*Prc**25*Pnc**5 + 735471*Prc**26*Pnc**4 + 863770*Prc**9*Pnc**20 + 35193430*Prc**10*Pnc**19 + 608653172*Prc**11*Pnc**18 + 5522950144*Prc**12*Pnc**17 + 28749060121*Prc**13*Pnc**16 + 86331965973*Prc**14*Pnc**15 + 128557437735*Prc**15*Pnc**14 - 18259584923*Prc**16*Pnc**13 - 460129599652*Prc**17*Pnc**12 - 932846913385*Prc**18*Pnc**11 - 1018117867324*Prc**19*Pnc**10 - 695738523421*Prc**20*Pnc**9 - 309721646328*Prc**21*Pnc**8 - 90091369512*Prc**22*Pnc**7 - 16756474524*Prc**23*Pnc**6 - 1892461522*Prc**24*Pnc**5 - 116396280*Prc**25*Pnc**4 - 2941884*Prc**26*Pnc**3 - 9465650*Prc**9*Pnc**19 - 363978904*Prc**10*Pnc**18 - 5766973856*Prc**11*Pnc**17 - 47625651705*Prc**12*Pnc**16 - 224494894862*Prc**13*Pnc**15 - 605959736948*Prc**14*Pnc**14 - 798049034901*Prc**15*Pnc**13 + 154674046636*Prc**16*Pnc**12 + 2419996996463*Prc**17*Pnc**11 + 4297268810047*Prc**18*Pnc**10 + 4127433336447*Prc**19*Pnc**9 + 2477973546646*Prc**20*Pnc**8 + 965986436950*Prc**21*Pnc**7 + 245053458888*Prc**22*Pnc**6 + 39565867286*Prc**23*Pnc**5 + 3859636174*Prc**24*Pnc**4 + 204013260*Prc**25*Pnc**3 + 4412826*Prc**26*Pnc**2 + 66565072*Prc**9*Pnc**18 + 2385757147*Prc**10*Pnc**17 + 34438990318*Prc**11*Pnc**16 + 257654347829*Prc**12*Pnc**15 + 1093908653848*Prc**13*Pnc**14 + 2638030772931*Prc**14*Pnc**13 + 3049197173098*Prc**15*Pnc**12 - 722912043180*Prc**16*Pnc**11 - 7727839555788*Prc**17*Pnc**10 - 11908065458697*Prc**18*Pnc**9 - 9950494318108*Prc**19*Pnc**8 - 5174558371874*Prc**20*Pnc**7 - 1735039042008*Prc**21*Pnc**6 - 375022797082*Prc**22*Pnc**5 - 50945940544*Prc**23*Pnc**4 - 4109881716*Prc**24*Pnc**3 - 175233960*Prc**25*Pnc**2 - 2941884*Prc**26*Pnc - 336457420*Prc**9*Pnc**17 - 11109808103*Prc**10*Pnc**16 - 145338555659*Prc**11*Pnc**15
    violationProb += -979920115149*Prc**12*Pnc**14 - 3724788538559*Prc**13*Pnc**13 - 7970684783619*Prc**14*Pnc**12 - 8015297082743*Prc**15*Pnc**11 + 2177967059253*Prc**16*Pnc**10 + 16708004336984*Prc**17*Pnc**9 + 22092326295980*Prc**18*Pnc**8 + 15836707762174*Prc**19*Pnc**7 + 7008137093805*Prc**20*Pnc**6 + 1974974473637*Prc**21*Pnc**5 + 352504971708*Prc**22*Pnc**4 + 38529281916*Prc**23*Pnc**3 + 2399479144*Prc**24*Pnc**2 + 73227330*Prc**25*Pnc + 735471*Prc**26 + 1297129224*Prc**9*Pnc**16 + 39063090551*Prc**10*Pnc**15 + 460591535371*Prc**11*Pnc**14 + 2782076232690*Prc**12*Pnc**13 + 9403795471954*Prc**13*Pnc**12 + 17717937899392*Prc**14*Pnc**11 + 15345649847998*Prc**15*Pnc**10 - 4566888512700*Prc**16*Pnc**9 - 25811192872492*Prc**17*Pnc**8 - 28885261062495*Prc**18*Pnc**7 - 17453471075572*Prc**19*Pnc**6 - 6423715201035*Prc**20*Pnc**5 - 1474972703964*Prc**21*Pnc**4 - 207921419988*Prc**22*Pnc**3 - 17058936212*Prc**23*Pnc**2 - 724335898*Prc**24*Pnc - 11767536*Prc**25 - 3952157716*Prc**9*Pnc**15 - 107610400230*Prc**10*Pnc**14 - 1136816475324*Prc**11*Pnc**13 - 6110132054886*Prc**12*Pnc**12 - 18224465310848*Prc**13*Pnc**11 - 29963295170584*Prc**14*Pnc**10 - 22091481584488*Prc**15*Pnc**9 + 6948248538281*Prc**16*Pnc**8 + 29290573075467*Prc**17*Pnc**7 + 27266252864043*Prc**18*Pnc**6 + 13575051938005*Prc**19*Pnc**5 + 4029383230576*Prc**20*Pnc**4 + 721259013872*Prc**21*Pnc**3 + 74950147400*Prc**22*Pnc**2 + 4078239990*Prc**23*Pnc + 87766206*Prc**24 + 9738063166*Prc**9*Pnc**14 + 237872303067*Prc**10*Pnc**13 + 2236411804206*Prc**11*Pnc**12 + 10613500645392*Prc**12*Pnc**11 + 27686647547198*Prc**13*Pnc**10 + 39308945147694*Prc**14*Pnc**9 + 24332479131086*Prc**15*Pnc**8 - 7841077947484*Prc**16*Pnc**7 - 24700726696948*Prc**17*Pnc**6 - 18701878253489*Prc**18*Pnc**5 - 7440734011944*Prc**19*Pnc**4 - 1704696373556*Prc**20*Pnc**3 - 222132491516*Prc**21*Pnc**2 - 15045711450*Prc**22*Pnc - 404189280*Prc**23 - 19708003114*Prc**9*Pnc**13 - 428620240174*Prc**10*Pnc**12 - 3558840576061*Prc**11*Pnc**11 - 14780254093331*Prc**12*Pnc**10 - 33372787603913*Prc**13*Pnc**9 - 40413196640008*Prc**14*Pnc**8 - 20656736877853*Prc**15*Pnc**7 + 6619977124913*Prc**16*Pnc**6 + 15463781246993*Prc**17*Pnc**5
    violationProb += 9238454055360*Prc**18*Pnc**4 + 2811730479316*Prc**19*Pnc**3 + 464450953304*Prc**20*Pnc**2 + 38977880942*Prc**21*Pnc + 1282393980*Prc**22 + 33106284800*Prc**9*Pnc**12 + 635945351970*Prc**10*Pnc**11 + 4621704878458*Prc**11*Pnc**10 + 16622955275160*Prc**12*Pnc**9 + 32090091161977*Prc**13*Pnc**8 + 32652644986563*Prc**14*Pnc**7 + 13502942322535*Prc**15*Pnc**6 - 4170055283135*Prc**16*Pnc**5 - 7089657634284*Prc**17*Pnc**4 - 3200230909612*Prc**18*Pnc**3 - 696146012276*Prc**19*Pnc**2 - 73378154850*Prc**20*Pnc - 2957574048*Prc**21 - 46472756148*Prc**9*Pnc**11 - 781405843341*Prc**10*Pnc**10 - 4918706196984*Prc**11*Pnc**9 - 15132490639627*Prc**12*Pnc**8 - 24607134860416*Prc**13*Pnc**7 - 20653373454438*Prc**14*Pnc**6 - 6733488879275*Prc**15*Pnc**5 + 1931359683582*Prc**16*Pnc**4 + 2310285704496*Prc**17*Pnc**3 + 736448316604*Prc**18*Pnc**2 + 101364808974*Prc**19*Pnc + 5086517436*Prc**20 + 54707098362*Prc**9*Pnc**10 + 796766118188*Prc**10*Pnc**9 + 4290226175444*Prc**11*Pnc**8 + 11122123731499*Prc**12*Pnc**7 + 14956135840806*Prc**13*Pnc**6 + 10112836609977*Prc**14*Pnc**5 + 2512271942480*Prc**15*Pnc**4 - 637914049212*Prc**16*Pnc**3 - 505805577420*Prc**17*Pnc**2 - 100838439130*Prc**18*Pnc - 6553898208*Prc**19 - 54037410538*Prc**9*Pnc**9 - 673216469565*Prc**10*Pnc**8 - 3054372331940*Prc**11*Pnc**7 - 6549642788140*Prc**12*Pnc**6 - 7113726731392*Prc**13*Pnc**5 - 3755013635432*Prc**14*Pnc**4 - 677858200668*Prc**15*Pnc**3 + 141969096984*Prc**16*Pnc**2 + 66532945050*Prc**17*Pnc + 6193483010*Prc**18 + 44677410658*Prc**9*Pnc**8 + 468944746435*Prc**10*Pnc**7 + 1759463464915*Prc**11*Pnc**6 + 3047634434224*Prc**12*Pnc**5 + 2590907563884*Prc**13*Pnc**4 + 1021238585796*Prc**14*Pnc**3 + 124658014052*Prc**15*Pnc**2 - 19031623182*Prc**16*Pnc - 3959429760*Prc**17 - 30740068882*Prc**9*Pnc**7 - 266760725275*Prc**10*Pnc**6 - 808135235812*Prc**11*Pnc**5 - 1095543759324*Prc**12*Pnc**4 - 697351963104*Prc**13*Pnc**3 - 191656636600*Prc**14*Pnc**2 - 13944920990*Prc**15*Pnc + 1156895883*Prc**16 + 17428289388*Prc**9*Pnc**6 + 122062383294*Prc**10*Pnc**5 + 289207368536*Prc**11*Pnc**4 + 293452488292*Prc**12*Pnc**3 + 130602400108*Prc**13*Pnc**2 + 22148272146*Prc**14*Pnc + 713897184*Prc**15
    violationProb += -8018267860*Prc**9*Pnc**5 - 43889540451*Prc**10*Pnc**4 - 77754229876*Prc**11*Pnc**3 - 55127947480*Prc**12*Pnc**2 - 15181507110*Prc**13*Pnc - 1185579252*Prc**14 + 2924434452*Prc**9*Pnc**4 + 11955705464*Prc**10*Pnc**3 + 14778209956*Prc**11*Pnc**2 + 6477296650*Prc**12*Pnc + 823727520*Prc**13 - 815302796*Prc**9*Pnc**3 - 2321724646*Prc**10*Pnc**2 - 1770609126*Prc**11*Pnc - 357929220*Prc**12 + 163529732*Prc**9*Pnc**2 + 286620510*Prc**10*Pnc + 100558944*Prc**11 - 21047972*Prc**9*Pnc - 16915833*Prc**10 + 1307504*Prc**9
    return violationProb

