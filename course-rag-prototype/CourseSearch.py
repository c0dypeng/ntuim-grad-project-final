from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class CourseSearch(BaseModel):
    """Search over a database of courses in a university."""

    course_name: Optional[str] = Field(
        None,
        description="Name of the course. 課程名稱"
    )
    course_classNumber: Optional[str] = Field(
        None,
        description="Class number of the course. 課程班號"
    )
    course_credit: Optional[float] = Field(
        None,
        description="Credit value of the course. 課程學分"
    )
    course_day: Optional[float] = Field(
        None,
        description="Day of the course. 1 is for Monday, 2 is for Tuesday, .... 開課日子"
    )
    course_id: Optional[str] = Field(
        None,
        description="ID of the course. 課程代碼"
    )
    course_identifier: Optional[str] = Field(
        None,
        description="Identifier of the course."
    )
    course_number: Optional[str] = Field(
        None,
        description="Number of the course."
    )
    course_period: Optional[str] = Field(
        None,
        description="Period of the course. 1, 2, 3, and 4 are in the morning, 5 to 10 are in the afternoon, and A to D are in the evening."
    )
    course_school: Optional[str] = Field(
        None,
        description="School offering the course. 開課學校"
    )
    course_college : Optional[str] = Field(
        None,
        description="College offering the course. 開課學院"
    )
    course_department: Optional[str] = Field(
        None,
        description="Department offering the course. 開課系所"
    )
    course_semester: Optional[str] = Field(
        None,
        description="Semester of the course. 開課學期"
    )
    course_signMethod: Optional[float] = Field(
        None,
        description="Sign method of the course. 加簽方法"
    )
    course_teacher: Optional[str] = Field(
        None,
        description="Teacher of the course. 授課老師"
    )
    

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

    def getFilter(self):
        filter_dict = {}
        for field in self.__fields__:
            value = getattr(self, field)
            if value is not None and value != getattr(self.__fields__[field], "default", None):
                if isinstance(value, float):
                    value = int(value)
                filter_dict[field] = {"$eq": value}
        return filter_dict