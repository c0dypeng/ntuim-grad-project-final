from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

class College(str, Enum):
    文學院 = "文學院"
    共同教育中心 = "共同教育中心"
    理學院 = "理學院"
    生命科學院 = "生命科學院"
    工學院 = "工學院"
    法律學院 = "法律學院"
    生物資源暨農學院 = "生物資源暨農學院"
    醫學院 = "醫學院"
    管理學院 = "管理學院"

class Department(str, Enum):
    中國文學系 = "中國文學系"
    日本語文學系 = "日本語文學系"
    外國語文學系 = "外國語文學系"
    共同教育組 = "共同教育組"
    應用數學科學研究所 = "應用數學科學研究所"
    基因體與系統生物學學位學程 = "基因體與系統生物學學位學程"
    物理學系 = "物理學系"
    地質科學系 = "地質科學系"
    土木工程學系 = "土木工程學系"
    法律學系 = "法律學系"
    生物環境系統工程學系 = "生物環境系統工程學系"
    藥學系 = "藥學系"
    財務金融學系 = "財務金融學系"
    環境工程學研究所 = "環境工程學研究所"
    華語教學碩士學位學程 = "華語教學碩士學位學程"
    醫學檢驗暨生物技術學系 = "醫學檢驗暨生物技術學系"
    物理治療學系 = "物理治療學系"
    農業經濟學系 = "農業經濟學系"
    臨床醫學研究所 = "臨床醫學研究所"
    植物病理與微生物學系 = "植物病理與微生物學系"
    海洋研究所 = "海洋研究所"
    歷史學系 = "歷史學系"
    電機工程學系 = "電機工程學系"
    化學工程學系 = "化學工程學系"
    圖書資訊學系 = "圖書資訊學系"
    大氣科學系 = "大氣科學系"
    生物機電工程學系 = "生物機電工程學系"
    中等學校教育學程 = "中等學校教育學程"
    工程科學及海洋工程學系 = "工程科學及海洋工程學系"
    天文物理研究所 = "天文物理研究所"
    植物醫學碩士學位學程 = "植物醫學碩士學位學程"
    園藝暨景觀學系 = "園藝暨景觀學系"
    政治學系 = "政治學系"
    化學系 = "化學系"
    資訊工程學系 = "資訊工程學系"
    機械工程學系 = "機械工程學系"
    工商管理學系 = "工商管理學系"
    哲學系 = "哲學系"
    森林環境暨資源學系 = "森林環境暨資源學系"
    昆蟲學系 = "昆蟲學系"
    翻譯碩士學位學程 = "翻譯碩士學位學程"
    臨床牙醫學研究所 = "臨床牙醫學研究所"
    生物科技研究所 = "生物科技研究所"
    戲劇學系 = "戲劇學系"
    商學研究所 = "商學研究所"
    臨床藥學研究所 = "臨床藥學研究所"
    資訊管理學系 = "資訊管理學系"
    數學系 = "數學系"
    國際企業學系 = "國際企業學系"
    應用力學研究所 = "應用力學研究所"
    生態學與演化生物學研究所 = "生態學與演化生物學研究所"
    國際三校農業生技與健康醫療碩士學位學程 = "國際三校農業生技與健康醫療碩士學位學程"
    藝術史研究所 = "藝術史研究所"
    臨床動物醫學研究所 = "臨床動物醫學研究所"
    經濟學系 = "經濟學系"
    創業創新管理碩士在職專班 = "創業創新管理碩士在職專班"
    免疫學研究所 = "免疫學研究所"
    光電工程學研究所 = "光電工程學研究所"
    腫瘤醫學研究所 = "腫瘤醫學研究所"
    分子醫學研究所 = "分子醫學研究所"
    職能治療學系 = "職能治療學系"
    動物科學技術學系 = "動物科學技術學系"
    生理學研究所 = "生理學研究所"
    社會工作學系 = "社會工作學系"
    地理環境資源學系 = "地理環境資源學系"
    公共衛生學系 = "公共衛生學系"
    漁業科學研究所 = "漁業科學研究所"
    生化科學研究所 = "生化科學研究所"
    人類學系 = "人類學系"
    電子工程學研究所 = "電子工程學研究所"
    心理學系 = "心理學系"
    全球衛生碩士學位學程 = "全球衛生碩士學位學程"
    管理學院企業管理碩士專班_GMBA = "管理學院企業管理碩士專班(GMBA)"
    生醫電子與資訊學研究所 = "生醫電子與資訊學研究所"
    材料科學與工程學系 = "材料科學與工程學系"
    農業化學系 = "農業化學系"
    生物多樣性國際碩士學位學程 = "生物多樣性國際碩士學位學程"
    電信工程學研究所 = "電信工程學研究所"
    應用物理學研究所 = "應用物理學研究所"
    會計學系 = "會計學系"
    醫學工程學系 = "醫學工程學系"
    分子暨比較病理生物學研究所 = "分子暨比較病理生物學研究所"
    醫療器材與醫學影像研究所 = "醫療器材與醫學影像研究所"
    統計與數據科學研究所 = "統計與數據科學研究所"
    國際體育運動事務學士學位學程 = "國際體育運動事務學士學位學程"
    建築與城鄉研究所 = "建築與城鄉研究所"
    微生物學研究所 = "微生物學研究所"
    科際整合法律學研究所 = "科際整合法律學研究所"
    音樂學研究所 = "音樂學研究所"
    護理學系 = "護理學系"
    牙醫學系 = "牙醫學系"
    流行病學與預防醫學研究所 = "流行病學與預防醫學研究所"
    資訊網路與多媒體研究所 = "資訊網路與多媒體研究所"
    食品科技研究所 = "食品科技研究所"
    生化科技學系 = "生化科技學系"
    創新設計學院 = "創新設計學院"
    創新領域學士學位學程 = "創新領域學士學位學程"
    國家發展研究所 = "國家發展研究所"
    公共衛生碩士學位學程 = "公共衛生碩士學位學程"
    學士後護理學系 = "學士後護理學系"
    生命科學系 = "生命科學系"
    農藝學系 = "農藝學系"
    病理學研究所 = "病理學研究所"
    獸醫學系 = "獸醫學系"
    文學院 = "文學院"
    智慧工程科技全英語學士學位學程 = "智慧工程科技全英語學士學位學程"
    新聞研究所 = "新聞研究所"
    通識教育組 = "通識教育組"
    分子與細胞生物學研究所 = "分子與細胞生物學研究所"
    藥理學研究所 = "藥理學研究所"
    毒理學研究所 = "毒理學研究所"
    社會學系 = "社會學系"
    腦與心智科學研究所 = "腦與心智科學研究所"
    資料科學博士學位學程 = "資料科學博士學位學程"
    工業工程學研究所 = "工業工程學研究所"
    公共事務研究所 = "公共事務研究所"
    環境與職業健康科學研究所 = "環境與職業健康科學研究所"
    運動設施與健康管理碩士學位學程 = "運動設施與健康管理碩士學位學程"
    語言學研究所 = "語言學研究所"
    生物產業傳播暨發展學系 = "生物產業傳播暨發展學系"
    基因體暨蛋白體醫學研究所 = "基因體暨蛋白體醫學研究所"
    醫學系 = "醫學系"
    奈米工程與科學碩士學位學程 = "奈米工程與科學碩士學位學程"
    防災減害與韌性碩士學位學程 = "防災減害與韌性碩士學位學程"
    口腔生物科學研究所 = "口腔生物科學研究所"
    生物科技與食品營養學士學位學程 = "生物科技與食品營養學士學位學程"
    植物科學研究所 = "植物科學研究所"
    事業經營碩士在職學位學程 = "事業經營碩士在職學位學程"
    生物化學暨分子生物學研究所 = "生物化學暨分子生物學研究所"
    高階管理碩士專班_EMBA = "高階管理碩士專班(EMBA)"
    健康政策與管理研究所 = "健康政策與管理研究所"
    統計碩士學位學程 = "統計碩士學位學程"
    臺灣文學研究所 = "臺灣文學研究所"
    高分子科學與工程學研究所 = "高分子科學與工程學研究所"
    創意創業學程 = "創意創業學程"
    元件材料與異質整合碩士學位學程 = "元件材料與異質整合碩士學位學程"
    生物科技管理碩士在職學位學程 = "生物科技管理碩士在職學位學程"
    外語教學暨資源中心 = "外語教學暨資源中心"
    解剖學暨細胞生物學研究所 = "解剖學暨細胞生物學研究所"
    全球衛生博士學位學程 = "全球衛生博士學位學程"
    健康行為與社區科學研究所 = "健康行為與社區科學研究所"
    氣候變遷與永續發展國際博士學位學程 = "氣候變遷與永續發展國際博士學位學程"
    地球系統科學國際研究生博士學位學程 = "地球系統科學國際研究生博士學位學程"
    精準健康博士學位學程 = "精準健康博士學位學程"
    事業經營法務碩士在職學位學程 = "事業經營法務碩士在職學位學程"
    奈米工程與科學博士學位學程 = "奈米工程與科學博士學位學程"
    永續化學科技國際研究生博士學位學程 = "永續化學科技國際研究生博士學位學程"
    積體電路設計與自動化博士學位學程 = "積體電路設計與自動化博士學位學程"
    管理學院 = "管理學院"
    全球農業科技與基因體科學碩士學位學程 = "全球農業科技與基因體科學碩士學位學程"
    氣候變遷與永續發展國際學位學程 = "氣候變遷與永續發展國際學位學程"

class CourseSearch(BaseModel):
    """Search over a database of courses in a university."""

    上課地點: Optional[str] = Field(
        None,
        description="Location of the course."
    )
    上課星期: Optional[float] = Field(
        None,
        description="Day of the course. 1 is for Monday, 2 is for Tuesday, .... 開課日子"
    )
    上課節次: Optional[str] = Field(
        None,
        description="Period of the course. 0 is for 7:10-8:00, 1 is for 8:10-9:00, 2 is for 9:10-10:00, 3 is for 10:20-11:10, 4 is for 11:20-12:10, 5 is for 12:20-13:10, 6 is for 13:20-14:10, 7 is for 14:20-15:10, 8 is for 15:30-16:20, 9 is for 16:30-17:20, 10 is for 17:30-18:20, A is for 18:25-19:15, B is for 19:20-20:10, C is for 20:15-21:05, D is for 21:10-22:00."
    )
    學分: Optional[int] = Field(
        None,
        description="Credit of the course."
    )
    所屬學院: Optional[College] = Field(
        None,
        description="College of the course."
    )
    所屬系所: Optional[Department] = Field(
        None,
        description="Department of the course."
    )
    授課教師: Optional[str] = Field(
        None,
        description="Instructor of the course."
    )
    # 課程名稱: Optional[str] = Field(
    #     None,
    #     description="Name of the course."
    # )
    課程流水號: Optional[int] = Field(
        None,
        description="Unique identifier for the course."
    )
    通識領域: Optional[str] = Field(
        None,
        description="General education field of the course."
    )

    def pretty_print(self) -> None:
        for field in self.model_fields:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.model_fields[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

    def getFilter(self):
        filter_dict = {}
        for field in self.model_fields:
            value = getattr(self, field)
            if value is not None and value != getattr(self.model_fields[field], "default", None):
                if(field == "course_period"):
                    filter_dict[field] = {"$in": [value]}
                else:
                    filter_dict[field] = {"$eq": value}
        return filter_dict