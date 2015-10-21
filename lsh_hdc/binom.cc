/*
 * =====================================================================================
 *
 *       Filename:  ext.cc
 *
 *    Description:  C and C++ code relevant to LSH-HDC Python package
 *
 *        Version:  1.0
 *        Created:  10/21/2015 13:12:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Eugene Scherba (es), escherba@gmail.com
 *   Organization:  -
 *
 * =====================================================================================
 */
#include "binom.h"

unsigned long long bctable[] = {
    6ULL, 10ULL, 15ULL, 20ULL, 21ULL, 35ULL, 28ULL, 56ULL, 70ULL, 36ULL, 84ULL, 126ULL,
    45ULL, 120ULL, 210ULL, 252ULL, 55ULL, 165ULL, 330ULL, 462ULL, 66ULL, 220ULL,
    495ULL, 792ULL, 924ULL, 78ULL, 286ULL, 715ULL, 1287ULL, 1716ULL, 91ULL, 364ULL,
    1001ULL, 2002ULL, 3003ULL, 3432ULL, 105ULL, 455ULL, 1365ULL, 3003ULL, 5005ULL,
    6435ULL, 120ULL, 560ULL, 1820ULL, 4368ULL, 8008ULL, 11440ULL, 12870ULL, 136ULL,
    680ULL, 2380ULL, 6188ULL, 12376ULL, 19448ULL, 24310ULL, 153ULL, 816ULL, 3060ULL,
    8568ULL, 18564ULL, 31824ULL, 43758ULL, 48620ULL, 171ULL, 969ULL, 3876ULL,
    11628ULL, 27132ULL, 50388ULL, 75582ULL, 92378ULL, 190ULL, 1140ULL, 4845ULL,
    15504ULL, 38760ULL, 77520ULL, 125970ULL, 167960ULL, 184756ULL, 210ULL, 1330ULL,
    5985ULL, 20349ULL, 54264ULL, 116280ULL, 203490ULL, 293930ULL, 352716ULL, 231ULL,
    1540ULL, 7315ULL, 26334ULL, 74613ULL, 170544ULL, 319770ULL, 497420ULL,
    646646ULL, 705432ULL, 253ULL, 1771ULL, 8855ULL, 33649ULL, 100947ULL, 245157ULL,
    490314ULL, 817190ULL, 1144066ULL, 1352078ULL, 276ULL, 2024ULL, 10626ULL,
    42504ULL, 134596ULL, 346104ULL, 735471ULL, 1307504ULL, 1961256ULL, 2496144ULL,
    2704156ULL, 300ULL, 2300ULL, 12650ULL, 53130ULL, 177100ULL, 480700ULL,
    1081575ULL, 2042975ULL, 3268760ULL, 4457400ULL, 5200300ULL, 325ULL, 2600ULL,
    14950ULL, 65780ULL, 230230ULL, 657800ULL, 1562275ULL, 3124550ULL, 5311735ULL,
    7726160ULL, 9657700ULL, 10400600ULL, 351ULL, 2925ULL, 17550ULL, 80730ULL,
    296010ULL, 888030ULL, 2220075ULL, 4686825ULL, 8436285ULL, 13037895ULL,
    17383860ULL, 20058300ULL, 378ULL, 3276ULL, 20475ULL, 98280ULL, 376740ULL,
    1184040ULL, 3108105ULL, 6906900ULL, 13123110ULL, 21474180ULL, 30421755ULL,
    37442160ULL, 40116600ULL, 406ULL, 3654ULL, 23751ULL, 118755ULL, 475020ULL,
    1560780ULL, 4292145ULL, 10015005ULL, 20030010ULL, 34597290ULL, 51895935ULL,
    67863915ULL, 77558760ULL, 435ULL, 4060ULL, 27405ULL, 142506ULL, 593775ULL,
    2035800ULL, 5852925ULL, 14307150ULL, 30045015ULL, 54627300ULL, 86493225ULL,
    119759850ULL, 145422675ULL, 155117520ULL, 465ULL, 4495ULL, 31465ULL, 169911ULL,
    736281ULL, 2629575ULL, 7888725ULL, 20160075ULL, 44352165ULL, 84672315ULL,
    141120525ULL, 206253075ULL, 265182525ULL, 300540195ULL, 496ULL, 4960ULL,
    35960ULL, 201376ULL, 906192ULL, 3365856ULL, 10518300ULL, 28048800ULL,
    64512240ULL, 129024480ULL, 225792840ULL, 347373600ULL, 471435600ULL,
    565722720ULL, 601080390ULL, 528ULL, 5456ULL, 40920ULL, 237336ULL, 1107568ULL,
    4272048ULL, 13884156ULL, 38567100ULL, 92561040ULL, 193536720ULL, 354817320ULL,
    573166440ULL, 818809200ULL, 1037158320ULL, 1166803110ULL, 561ULL, 5984ULL,
    46376ULL, 278256ULL, 1344904ULL, 5379616ULL, 18156204ULL, 52451256ULL,
    131128140ULL, 286097760ULL, 548354040ULL, 927983760ULL, 1391975640ULL,
    1855967520ULL, 2203961430ULL, 2333606220ULL, 595ULL, 6545ULL, 52360ULL,
    324632ULL, 1623160ULL, 6724520ULL, 23535820ULL, 70607460ULL, 183579396ULL,
    417225900ULL, 834451800ULL, 1476337800ULL, 2319959400ULL, 3247943160ULL,
    4059928950ULL, 4537567650ULL, 630ULL, 7140ULL, 58905ULL, 376992ULL, 1947792ULL,
    8347680ULL, 30260340ULL, 94143280ULL, 254186856ULL, 600805296ULL,
    1251677700ULL, 2310789600ULL, 3796297200ULL, 5567902560ULL, 7307872110ULL,
    8597496600ULL, 9075135300ULL, 666ULL, 7770ULL, 66045ULL, 435897ULL, 2324784ULL,
    10295472ULL, 38608020ULL, 124403620ULL, 348330136ULL, 854992152ULL,
    1852482996ULL, 3562467300ULL, 6107086800ULL, 9364199760ULL, 12875774670ULL,
    15905368710ULL, 17672631900ULL, 703ULL, 8436ULL, 73815ULL, 501942ULL,
    2760681ULL, 12620256ULL, 48903492ULL, 163011640ULL, 472733756ULL,
    1203322288ULL, 2707475148ULL, 5414950296ULL, 9669554100ULL, 15471286560ULL,
    22239974430ULL, 28781143380ULL, 33578000610ULL, 35345263800ULL, 741ULL,
    9139ULL, 82251ULL, 575757ULL, 3262623ULL, 15380937ULL, 61523748ULL,
    211915132ULL, 635745396ULL, 1676056044ULL, 3910797436ULL, 8122425444ULL,
    15084504396ULL, 25140840660ULL, 37711260990ULL, 51021117810ULL,
    62359143990ULL, 68923264410ULL, 780ULL, 9880ULL, 91390ULL, 658008ULL,
    3838380ULL, 18643560ULL, 76904685ULL, 273438880ULL, 847660528ULL,
    2311801440ULL, 5586853480ULL, 12033222880ULL, 23206929840ULL, 40225345056ULL,
    62852101650ULL, 88732378800ULL, 113380261800ULL, 131282408400ULL,
    137846528820ULL, 820ULL, 10660ULL, 101270ULL, 749398ULL, 4496388ULL,
    22481940ULL, 95548245ULL, 350343565ULL, 1121099408ULL, 3159461968ULL,
    7898654920ULL, 17620076360ULL, 35240152720ULL, 63432274896ULL,
    103077446706ULL, 151584480450ULL, 202112640600ULL, 244662670200ULL,
    269128937220ULL, 861ULL, 11480ULL, 111930ULL, 850668ULL, 5245786ULL,
    26978328ULL, 118030185ULL, 445891810ULL, 1471442973ULL, 4280561376ULL,
    11058116888ULL, 25518731280ULL, 52860229080ULL, 98672427616ULL,
    166509721602ULL, 254661927156ULL, 353697121050ULL, 446775310800ULL,
    513791607420ULL, 538257874440ULL, 903ULL, 12341ULL, 123410ULL, 962598ULL,
    6096454ULL, 32224114ULL, 145008513ULL, 563921995ULL, 1917334783ULL,
    5752004349ULL, 15338678264ULL, 36576848168ULL, 78378960360ULL,
    151532656696ULL, 265182149218ULL, 421171648758ULL, 608359048206ULL,
    800472431850ULL, 960566918220ULL, 1052049481860ULL, 946ULL, 13244ULL,
    135751ULL, 1086008ULL, 7059052ULL, 38320568ULL, 177232627ULL, 708930508ULL,
    2481256778ULL, 7669339132ULL, 21090682613ULL, 51915526432ULL,
    114955808528ULL, 229911617056ULL, 416714805914ULL, 686353797976ULL,
    1029530696964ULL, 1408831480056ULL, 1761039350070ULL, 2012616400080ULL,
    2104098963720ULL, 990ULL, 14190ULL, 148995ULL, 1221759ULL, 8145060ULL,
    45379620ULL, 215553195ULL, 886163135ULL, 3190187286ULL, 10150595910ULL,
    28760021745ULL, 73006209045ULL, 166871334960ULL, 344867425584ULL,
    646626422970ULL, 1103068603890ULL, 1715884494940ULL, 2438362177020ULL,
    3169870830126ULL, 3773655750150ULL, 4116715363800ULL, 1035ULL, 15180ULL,
    163185ULL, 1370754ULL, 9366819ULL, 53524680ULL, 260932815ULL, 1101716330ULL,
    4076350421ULL, 13340783196ULL, 38910617655ULL, 101766230790ULL,
    239877544005ULL, 511738760544ULL, 991493848554ULL, 1749695026860ULL,
    2818953098830ULL, 4154246671960ULL, 5608233007146ULL, 6943526580276ULL,
    7890371113950ULL, 8233430727600ULL, 1081ULL, 16215ULL, 178365ULL, 1533939ULL,
    10737573ULL, 62891499ULL, 314457495ULL, 1362649145ULL, 5178066751ULL,
    17417133617ULL, 52251400851ULL, 140676848445ULL, 341643774795ULL,
    751616304549ULL, 1503232609098ULL, 2741188875414ULL, 4568648125690ULL,
    6973199770790ULL, 9762479679106ULL, 12551759587422ULL, 14833897694226ULL,
    16123801841550ULL, 1128ULL, 17296ULL, 194580ULL, 1712304ULL, 12271512ULL,
    73629072ULL, 377348994ULL, 1677106640ULL, 6540715896ULL, 22595200368ULL,
    69668534468ULL, 192928249296ULL, 482320623240ULL, 1093260079344ULL,
    2254848913647ULL, 4244421484512ULL, 7309837001104ULL, 11541847896480ULL,
    16735679449896ULL, 22314239266528ULL, 27385657281648ULL, 30957699535776ULL,
    32247603683100ULL, 1176ULL, 18424ULL, 211876ULL, 1906884ULL, 13983816ULL,
    85900584ULL, 450978066ULL, 2054455634ULL, 8217822536ULL, 29135916264ULL,
    92263734836ULL, 262596783764ULL, 675248872536ULL, 1575580702584ULL,
    3348108992991ULL, 6499270398159ULL, 11554258485616ULL, 18851684897584ULL,
    28277527346376ULL, 39049918716424ULL, 49699896548176ULL, 58343356817424ULL,
    63205303218876ULL, 1225ULL, 19600ULL, 230300ULL, 2118760ULL, 15890700ULL,
    99884400ULL, 536878650ULL, 2505433700ULL, 10272278170ULL, 37353738800ULL,
    121399651100ULL, 354860518600ULL, 937845656300ULL, 2250829575120ULL,
    4923689695575ULL, 9847379391150ULL, 18053528883775ULL, 30405943383200ULL,
    47129212243960ULL, 67327446062800ULL, 88749815264600ULL, 108043253365600ULL,
    121548660036300ULL, 126410606437752ULL, 1275ULL, 20825ULL, 249900ULL,
    2349060ULL, 18009460ULL, 115775100ULL, 636763050ULL, 3042312350ULL,
    12777711870ULL, 47626016970ULL, 158753389900ULL, 476260169700ULL,
    1292706174900ULL, 3188675231420ULL, 7174519270695ULL, 14771069086725ULL,
    27900908274925ULL, 48459472266975ULL, 77535155627160ULL, 114456658306760ULL,
    156077261327400ULL, 196793068630200ULL, 229591913401900ULL,
    247959266474052ULL, 1326ULL, 22100ULL, 270725ULL, 2598960ULL, 20358520ULL,
    133784560ULL, 752538150ULL, 3679075400ULL, 15820024220ULL, 60403728840ULL,
    206379406870ULL, 635013559600ULL, 1768966344600ULL, 4481381406320ULL,
    10363194502115ULL, 21945588357420ULL, 42671977361650ULL, 76360380541900ULL,
    125994627894135ULL, 191991813933920ULL, 270533919634160ULL,
    352870329957600ULL, 426384982032100ULL, 477551179875952ULL,
    495918532948104ULL, 1378ULL, 23426ULL, 292825ULL, 2869685ULL, 22957480ULL,
    154143080ULL, 886322710ULL, 4431613550ULL, 19499099620ULL, 76223753060ULL,
    266783135710ULL, 841392966470ULL, 2403979904200ULL, 6250347750920ULL,
    14844575908435ULL, 32308782859535ULL, 64617565719070ULL, 119032357903550ULL,
    202355008436035ULL, 317986441828055ULL, 462525733568080ULL,
    623404249591760ULL, 779255311989700ULL, 903936161908052ULL,
    973469712824056ULL, 1431ULL, 24804ULL, 316251ULL, 3162510ULL, 25827165ULL,
    177100560ULL, 1040465790ULL, 5317936260ULL, 23930713170ULL, 95722852680ULL,
    343006888770ULL, 1108176102180ULL, 3245372870670ULL, 8654327655120ULL,
    21094923659355ULL, 47153358767970ULL, 96926348578605ULL, 183649923622620ULL,
    321387366339585ULL, 520341450264090ULL, 780512175396135ULL,
    1085929983159840ULL, 1402659561581460ULL, 1683191473897752ULL,
    1877405874732108ULL, 1946939425648112ULL
};


unsigned long long binomial(unsigned int n, unsigned int k) {
    // Originally by Lee Daniel Crocker
    // http://etceterology.com/fast-binomial-coefficients
    //
    //assert(n >= 0 && k >= 0);

    if (0u == k || n == k) return 1ULL;
    if (k > n) return 0ULL;

    if (k > (n - k)) {
        k = n - k;
    }
    if (1u == k) return (unsigned long long)n;

    if (n <= 54u && k <= 54u) {
        return bctable[(((n - 3u) * (n - 3u)) >> 2u) + (k - 2u)];
    }
    /* Last resort: actually calculate */
    unsigned int i;
    long long b = 1LL;
    for (i = 1u; i <= k; ++i) {
        b *= (long long)((n - k) + i);
        if (b < 0) return (unsigned long long)-1LL; /* Overflow */
        b /= (long long)i;
    }
    return b;
}
