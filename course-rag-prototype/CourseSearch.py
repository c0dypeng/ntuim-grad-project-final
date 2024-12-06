from typing import Optional
from pydantic import BaseModel, Field

class CourseSearch(BaseModel):
    """Search over a database of courses in a university."""

    上課星期: Optional[float] = Field(
        None,
        description="Day of the course. 1 is for Monday, 2 is for Tuesday, .... 開課日子"
    )
    上課節次: Optional[str] = Field(
        None,
        description="Period of the course. 0 is for 7:10-8:00, 1 is for 8:10-9:00, 2 is for 9:10-10:00, 3 is for 10:20-11:10, 4 is for 11:20-12:10, 5 is for 12:20-13:10, 6 is for 13:20-14:10, 7 is for 14:20-15:10, 8 is for 15:30-16:20, 9 is for 16:30-17:20, 10 is for 17:30-18:20, A is for 18:25-19:15, B is for 19:20-20:10, C is for 20:15-21:05, D is for 21:10-22:00."
    )
    所屬系所: Optional[str] = Field(
        None,
        description="Department of the course."
    )
    授課教師: Optional[str] = Field(
        None,
        description="Instructor of the course."
    )
    課程流水號: Optional[int] = Field(
        None,
        description="Unique identifier for the course."
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
                # if isinstance(value, float):
                #     value = int(value)
                if(field == "course_period"):
                    filter_dict[field] = {"$in": [value]}
                else:
                    filter_dict[field] = {"$eq": value}
        return filter_dict