from llama_index.core import PromptTemplate


class PromptTemplates:
    """提示模板管理类"""

    @staticmethod
    def get_default_qa_template() -> str:
        """获取默认问答提示模板"""
        return (
            "上下文信息如下所示。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请根据上述上下文信息回答用户的问题。如果你不知道答案，请说你不知道。\n"
            "问题: {query_str}\n"
            "回答: "
        )

    @staticmethod
    def get_default_refine_template() -> str:
        """获取默认精炼提示模板"""
        return (
            "这是原始问题: {context_str}\n"
            "我们已经提供了一些相关的上下文信息: {existing_answer}\n"
            "---------------------\n"
            "{query_str}\n"
            "---------------------\n"
            "请根据新的上下文信息优化已有的回答。"
            "如果新的上下文信息没有用处，请沿用之前的回答。\n"
            "优化后的回答: "
        )

    @staticmethod
    def get_condense_question_template() -> str:
        """获取问题压缩提示模板"""
        return (
            "给定以下对话历史和后续问题，"
            "请将后续问题改写为一个独立的问题，包含所有必要的上下文信息。"
            "不要在改写后的问题中添加任何其他内容。\n"
            "---------------------\n"
            "聊天历史: {context_str}\n"
            "---------------------\n"
            "后续问题: {query_str}\n"
            "独立问题:"
        )

    @staticmethod
    def get_technical_qa_template() -> str:
        """获取技术问答提示模板"""
        return (
            "你是一个专业的技术助手。根据以下上下文信息回答问题：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请用专业术语回答以下问题。如果上下文信息不足以回答问题，请说明原因。\n"
            "问题: {query_str}\n"
            "专业回答: "
        )

    @staticmethod
    def get_concise_qa_template() -> str:
        """获取简洁问答提示模板"""
        return (
            "根据以下上下文信息，简洁地回答问题：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "问题: {query_str}\n"
            "简答: "
        )

    @staticmethod
    def get_creative_qa_template() -> str:
        """获取创意思维提示模板"""
        return (
            "你是一个富有创造力的助手。请用创新和独特的方式回答以下问题：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请发挥你的想象力，提供新颖的观点和解决方案。\n"
            "问题: {query_str}\n"
            "创意回答: "
        )

    @staticmethod
    def get_step_by_step_template() -> str:
        """获取步骤化解答提示模板"""
        return (
            "你是一个善于分解问题的专家。请按步骤详细解答以下问题：\n"
            "---------------------\n"
            "{my_context_str}\n"
            "---------------------\n"
            "请将解答过程分解为清晰的步骤，并解释每一步。\n"
            "问题: {my_query_str}\n"
            "解答步骤:\n"
            "1. "
        )

    @staticmethod
    def create_prompt_template(template_str: str, template_var_mappings: dict = None):
        """
        创建提示模板

        Args:
            template_str: 模板字符串
            template_var_mappings: 模板变量映射字典

        Returns:
            PromptTemplate: 创建的提示模板对象
        """
        try:
            if template_var_mappings:
                return PromptTemplate(template_str, template_var_mappings=template_var_mappings)
            else:
                return PromptTemplate(template_str)
        except Exception as e:
            raise

    @staticmethod
    def get_template_by_name( template_name: str) -> str:
        """
        根据模板名称获取模板字符串

        Args:
            template_name: 模板名称

        Returns:
            str: 模板字符串
        """
        template_map = {
            'technical': PromptTemplates.get_technical_qa_template(),
            'step_by_step': PromptTemplates.get_step_by_step_template(),
            'default': PromptTemplates.get_default_qa_template(),
            'concise': PromptTemplates.get_concise_qa_template(),
            'creative': PromptTemplates.get_creative_qa_template(),
            'condense': PromptTemplates.get_condense_question_template()
        }

        return template_map.get(template_name, PromptTemplates.get_default_qa_template())
