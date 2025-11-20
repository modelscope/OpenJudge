# -*- coding: utf-8 -*-
import asyncio

from rm_gallery.core.model.openai_llm import OpenAIChatModel

if __name__ == "__main__":
    model = OpenAIChatModel(
        model_name="qwen-plus",
        stream=False,
    )

    response = asyncio.run(
        model.achat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": """What is your name?
    Please response as
    ```json
    {{
        "name": "your name"
    }}
    ```
    """,
                },
            ],
            structured_model={"type": "json_object"},
        ),
    )

    print(response)
