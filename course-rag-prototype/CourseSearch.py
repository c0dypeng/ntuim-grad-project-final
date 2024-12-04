from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class CourseSearch(BaseModel):
    """Search over a database of courses in a university."""

    course_credit: Optional[float] = Field(
        None,
        description="Credit value of the course. 課程學分"
    )
    course_day: Optional[float] = Field(
        None,
        description="Day of the course. 1 is for Monday, 2 is for Tuesday, .... 開課日子"
    )
    course_period: Optional[str] = Field(
        None,
        description="Period of the course. 1, 2, 3, and 4 are in the morning, 5 to 10 are in the afternoon, and A to D are in the evening."
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
                # if isinstance(value, float):
                #     value = int(value)
                if(field == "course_period"):
                    filter_dict[field] = {"$in": [value]}
                else:
                    filter_dict[field] = {"$eq": value}
        return filter_dict