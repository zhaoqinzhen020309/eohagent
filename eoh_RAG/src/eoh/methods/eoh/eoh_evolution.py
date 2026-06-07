import re
import time
from ...llm.interface_LLM import InterfaceLLM


class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):

        # set prompt interface
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, llm_use_local, llm_local_url,
                                          self.debug_mode)

    def get_prompt_i1_with_rag(self, rag_guidance=None):
        base_prompt = self.prompt_task + "\n" \
                                         "First, describe your new algorithm and main steps in one sentence. \
                                       The description must be inside a brace. Next, implement it in Python as a function named \
                                       " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                      + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                      + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                      + self.prompt_other_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        final_prompt = base_prompt + rag_section + "Do not give additional explanations."
        return final_prompt

    def get_prompt_e1_with_rag(self, indivs, rag_guidance=None):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        base_prompt = self.prompt_task + "\n" \
                                         "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                      + prompt_indiv + \
                      "Please help me create a new algorithm that has a totally different form from the given ones. \n"

        instruction_prompt = "First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                             + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                             + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                             + self.prompt_other_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        return base_prompt + rag_section + instruction_prompt + "Do not give additional explanations."

    def get_prompt_e2_with_rag(self, indivs, rag_guidance=None):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        base_prompt = self.prompt_task + "\n" \
                                         "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                      + prompt_indiv + \
                      "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n" \
                      "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
                      The description must be inside a brace. Thirdly, implement it in Python as a function named \
                      " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                      + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                      + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                      + self.prompt_other_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        return base_prompt + rag_section + "Do not give additional explanations."

    def get_prompt_m1_with_rag(self, indiv1, rag_guidance=None):
        base_prompt = self.prompt_task + "\n" \
                                         "I have one algorithm with its code as follows. \
                                       Algorithm description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"

        instruction_prompt = "First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                             + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                             + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                             + self.prompt_other_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        return base_prompt + rag_section + instruction_prompt + "Do not give additional explanations."

    def get_prompt_m2_with_rag(self, indiv1, rag_guidance=None):
        base_prompt = self.prompt_task + "\n" \
                                         "I have one algorithm with its code as follows. \
                                       Algorithm description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"

        instruction_prompt = "First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
" + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                             + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                             + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                             + self.prompt_other_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        return base_prompt + rag_section + instruction_prompt + "Do not give additional explanations."

    def get_prompt_m3_with_rag(self, indiv1, rag_guidance=None):
        base_prompt = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n" + indiv1['code'] + "\n" \
                      + self.prompt_inout_inf + "\n"

        if rag_guidance:
            rag_section = f"""

{rag_guidance}

"""
        else:
            rag_section = ""

        return base_prompt + rag_section + "Do not give additional explanations."

    def _get_alg(self, prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        if len(algorithm) > 0:
            algorithm = algorithm[0]
        else:
            algorithm = "Unknown algorithm"

        if len(code) > 0:
            code = code[0]
        else:
            code = "# Error: no valid code generated"

        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def i1(self, rag_guidance=None):

        prompt_content = self.get_prompt_i1_with_rag(rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents, rag_guidance=None):

        prompt_content = self.get_prompt_e1_with_rag(parents, rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents, rag_guidance=None):

        prompt_content = self.get_prompt_e2_with_rag(parents, rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents, rag_guidance=None):

        prompt_content = self.get_prompt_m1_with_rag(parents, rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents, rag_guidance=None):

        prompt_content = self.get_prompt_m2_with_rag(parents, rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m3(self, parents, rag_guidance=None):

        prompt_content = self.get_prompt_m3_with_rag(parents, rag_guidance)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]