import torch
from torch import nn

class LoraModel(nn.Module):
    def __init__(self,model,config,adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name,self.peft_config[adapter_name])
    
    def add_adapter(self,adapter_name,config=None):
        if config is not None:
            # 含义：去 self.model 里找叫 "config" 的属性。
            # 如果找不到（有些野鸡模型可能没写config），就返回默认字典 {"model_type": "custom"}
            model_config = getattr(self.model,"config",{"model_type":"custom"})
        # 有些 config 是对象，有些是字典。
        # 如果是对象（有 to_dict 方法），就把它转成字典方便处理。
            if hasattr(model_config,"to_dict"):
               model_config = model_config.to_dict()
            #假设的函数，用来准备最终的lora配置
            config = self._prepare_lora_config(config, model_config)
            #把处理好的配置存进字典里
            self.peft_config[adapter_name] = config
        #的作用是遍历模型，把 Linear 层换成 LoraLayer   
        self._find_and_replace(adapter_name)
        #想象 bias 是方向盘。如果装了两个 LoRA (比如一个自动驾驶，一个辅助泊车)，
        # 它们都想抢方向盘(bias)，那就乱套了。
        # 所以这里规定：如果有多于1个适配器，且大家都不能动 bias，直接报错
        if len(self.peft_config > 1) and self.peft_config[adapter_name].bias != None:
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )
        # 3. 设置可训练参数
        # 这个函数会把大模型冻结(False)，把 LoRA 层解冻(True)
        mark_only_lora_as_trainable(self.model,self.peft_config[adapter_name].bias)

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model,adapter_name)
#_find_and_replace 会把模型里的层“拆下来换掉”。这就是那个“换上去的新零件
class Linear(nn.Linear,LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights",True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        #冻结权重
        self.weight.requires_grad = False
        #有些模型权重反过来的，如gpt2和conv1d类型的层
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        # 1. 重置参数
        nn.Linear.reset_parameters(self)

        # 2. 创建 LoRA 矩阵
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        # 3. 设置当前开关
        self.active_adapter = adapter_name
        #也是用来适配gpt2的现在没啥用了
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    #*args：负责回收那些没有名字的参数（按顺序排队）。
    #**kwargs：负责回收那些有名字的参数（带标签的）。
    def forward(self,x:torch.Tensor,*args,**kwargs):
        result = nn.Linear.forward(self, x) # 调用亲妈 nn.Linear 的计算
        
        # 2. 如果当前没把 LoRA 关掉 (disable_adapters=False)
        if not self.disable_adapters:
            # 3. 走旁路：计算 LoRA 的增量
            
            # 第一步：过 Dropout (防止过拟合)
            dropout_output = self.lora_dropout[self.active_adapter](x)
            
            # 第二步：过矩阵 A (降维) -> 得到一个很小的向量
            matrix_a_output = self.lora_A[self.active_adapter](dropout_output)
            
            # 第三步：过矩阵 B (升维) -> 变回原来的大小
            matrix_b_output = self.lora_B[self.active_adapter](matrix_a_output)
            
            # 第四步：乘以缩放系数 (Scaling)
            scaled_output = matrix_b_output * self.scaling[self.active_adapter]
            
            # 4. 【殊途同归】：把旁路的结果加到主路结果上
            result += scaled_output
            
        return result
