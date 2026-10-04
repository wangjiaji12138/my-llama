#输入输出结构
import base64
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing_extensions import Annotated

# 身份信息
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    
# 内置工具
class BuiltinTool(Enum):
    # 成员名-成员值
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"
    
# 原始的简单数据类型
Primitive = Union[str, int, float, bool, None]

# 递归数据类型
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]

class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    arguments: Union[str,Dict[str,RecursiveType]]
    arguments_json: Optional[str] = None
    
    # tool_name的字段校验器
    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v) # 按值查找，哪个成员的值是这个字符串
            except ValueError:
                return v
        return v
    
class ToolPromptFormat(Enum):
    """
    :cvar json:{
        "type":"function",
        "function:{
            "name": "function_name",
            "description": "function_description",
            "parameters": {...}
        }
        "
    }
    
    :cvar function_tag: <function=function_name>(parameters)</function>
    
    :cvar python_list: ["function_name(param1, param2)", ["function_name(param1, param2)"]
    """
    
    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"
    
class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"
    
class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = False
    default: Optional[Any] = None
    
class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None
    """
    例子:
    weather_tool = ToolDefinition(
        tool_name="get_weather",
        description="查询指定城市的当前天气",
        parameters={
            "city": ToolParamDefinition(type="string", description="城市名", required=True),
        }
    )   
    """

    @field_validator("tool_name",mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v
    
class RawMediaItem(BaseModel):
    type: Literal["image"] = "image" # 只能是image, 否则报错
    data: bytes | BytesIO # 文件里的bytes（不可读写）/ 内存里的字节流（可读写）
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 序列化:JSON不识别图片的二进制格式，只识别文本，需要将图片存进JSON时需要套壳
    @field_serializer("data")
    def serialize_data(self, data:Optional[bytes], _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")
    
    # 反序列化：从JSON中取出图片时反序列化成二进制文件
    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v
    
class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str

RawContentItem = Annotated[Union[RawTextItem, RawMediaItem], Field(discriminator = "type")]

RawContent = str | RawContentItem | List[RawContentItem]

class RawMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: RawContent
    
    context: Optional[RawContent] = None
    
    stop_reason: Optional[StopReason] = None
    tool_calls: List[ToolCall] = Field(default_factory=list) # 每个实例都拿到一个独立的新列表
    
class GenerationResult(BaseModel):
    token: int
    text: str
    logprobs: Optional[List[float]] = None
    
    source: Literal["input", "output"]
    
    batch_idx: int
    finished: bool
    ignore_token: bool
    
class QuantizationMode(str, Enum):
    none = "none"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"