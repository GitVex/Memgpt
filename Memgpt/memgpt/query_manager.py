"""
Query Manager
"""
import logging
from datetime import datetime
import uuid

import pinecone
import openai

class QueryManager:
    """
    Manages querying and interaction with OpenAI GPT-4 and Pinecone services.

    This class provides an interface for querying a Pinecone index, generating query embeddings and prompts for GPT-4, and managing interaction history.

    Usage
    -----
    >>> from memgpt.query_manager import QueryManager
    >>> qManager = QueryManager(params)
    >>> qManager.set_query('Your query goes here')
    >>> qManager.query_index() # Pinecone Similarity Search
    >>> qManager.generate_reposonse() # OpenAI Chat Completion

    Attributes
    ----------
    DEFAULT_PREFACE : str
        The default preface message for a generated GPT-4 prompt.
    user_id : str
        The ID of the user interacting with the instance.
    openai_api_key : str
        The OpenAI API key for OpenAI services.
    pinecone_api_key : str
        The Pinecone API key for Pinecone services.
    index_name : str
        The name of the Pinecone index to be queried.
    debug : bool, optional
        A flag indicating whether to enable debugging mode (default is False).
    pinecone_environment : str, optional
        The environment of the Pinecone service (default is 'asia-southeast1-gcp-free').
    context : list
        A list of dictionaries representing the most relevant matches for the current query.
    query : str
        The current query of the instance.
    query_embed : list
        The embedding of the current query, generated by OpenAI's "text-embedding-ada-002" model.
    prompt : list
        A list of dictionaries representing the generated prompt for GPT-4 based on the instance's context and query.
    index : Pinecone Index
        The Pinecone index associated with the instance.

    Methods
    -------
    get_status():
        Returns the current status of the instance as a dictionary.
    status():
        Prints the current status of the instance in a human-readable format.
    set_query(query):
        Sets the query attribute for the instance.
    __embed_query():
        Converts the current query into an embedding using the OpenAI's "text-embedding-ada-002" model.
    __rescore_matches(matches, top_k=5, threshold=0.91, role_weight=1.2, length_weight=1.2, semantic_bias=0):
        Rescores, prunes, and sorts a list of matches based on specific criteria.
    query_index(query=None, top_k=5, threshold=0.9, role_weight=1.2, length_weight=1.2, semantic_bias=0, rescoring=True):
        Queries the instance's index with a specified query and optionally rescores the matches.
    generate_prompt(preface_message=DEFAULT_PREFACE):
        Generates prompt for GPT-4 response based on the instance's context.
    generate_response():
        Generates a response to the instance's current query.
    """

    DEFAULT_PREFACE = 'You are a helpful assistant augmented with Quasi-Long-Term-Memory through a Pinecone index of previous conversations. \
    Answer the users questions to the best of your ability, and if you do not know the answer, say so and ask the user to teach you.'

    def __init__(self, user_id, openai_api_key, pinecone_api_key, index_name, debug=False, pinecone_environment="asia-southeast1-gcp-free"):
        """
        placeholder
        """

        self.user_id = user_id
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.debug = debug
        self.pinecone_environment = pinecone_environment

        self.context = []
        self.query = ''
        self.query_embed = []
        self.prompt = []
        self.response = {}

        pinecone.init(api_key=self.pinecone_api_key,
                      environment=self.pinecone_environment)
        self.index = pinecone.Index(self.index_name)

        openai.api_key = self.openai_api_key

        if self.debug:
            logging.basicConfig(
                filename=f'./debug/debugging_query_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log', level=logging.DEBUG)

    def get_status(self):
        """
        Returns the current status of the instance.

        This method builds a dictionary reflecting the current status of the instance, including the current values of 'user_id', 'index', 'debug', 'query', and 'context' attributes. If 'query' or 'context' is not set, it specifically states that these attributes are not yet set.

        Returns
        -------
        dict
            A dictionary representing the current status of the instance.
        """
        status_dict = {
            "user_id": self.user_id,
            "index": self.index,
            "debug": self.debug,
            "query": self.query if self.query != '' else "Not set yet",
            "context": self.context if self.context else "Not set yet",
            "prompt": self.prompt if self.prompt else "Not set yet",
            "response": self.response if self.response else "Not set yet"
        }

        return status_dict

    def status(self):
        """
        Prints the current status of the instance.

        This method retrieves the current status of the instance as a dictionary and prints it in a human-readable format.

        Returns
        -------
        None
        """
        status_dict = self.get_status()

        for key, value in status_dict.items():
            print(f"{key}: {value}")

    def set_query(self, query):
        """
        Sets the query attribute for the instance.

        This method assigns the input query to the 'query' attribute of the instance. 

        Parameters
        ----------
        query : str
            The query to be set for the instance.

        Returns
        -------
        None
        """

        self.query = query

    def __embed_query(self):
        """
        Converts the current query into an embedding using the OpenAI's "text-embedding-ada-002" model.

        This private method first performs input validation, checking whether the query is not empty and it is appropriately stripped off trailing whitespaces, newlines, and other impurities. It then sends the query to the OpenAI embedding creation service and performs validation on the response. If successful, the method assigns the first embedding from the returned data to the 'query_embed' attribute of the instance.

        Raises
        ------
        ValueError
            If the query is empty.

        TypeError
            If the response from the OpenAI service is not a dictionary or the 'data' field in the response is not a list.

        KeyError
            If the 'data' key is not present in the response from the OpenAI service.

        Returns
        -------
        None
        """

        if self.query == '':
            raise ValueError("Query is empty")
        
        if self.query is None:
            raise ValueError("Query is None")

        # remove trailing whitespaces, newlines and other impurities
        self.query = self.query.strip()

        response = openai.Embedding.create(
            input=self.query, model="text-embedding-ada-002")

        if not isinstance(response, dict):
            raise TypeError(
                f"[OpenAI] Expected response to be a dict, but got {type(response)}")

        if 'data' not in response:
            raise KeyError(
                f"[OpenAI] Expected 'data' key in response, but got {response.keys()}")

        res_data = response['data']

        if not isinstance(res_data, list):
            raise TypeError(
                f"[OpenAI] Expected 'data' to be a list, but got {type(res_data)}")

        embedding = res_data[0]['embedding']

        self.query_embed = embedding

    def __rescore_matches(self, matches, top_k=5, threshold=0.91, role_weight=1.2, length_weight=1.2, semantic_bias=0):
        """
        Rescores, prunes, and sorts a list of matches based on specific criteria.

        This private method first recalculates scores based on match 'role' and 'content' length, using provided weights and a semantic bias. 
        It then prunes matches falling below a specified score threshold, sorts remaining matches by their scores in descending order, and returns the top k matches.

        Parameters
        ----------
        matches : list[dict]
            The list of matches to be rescored. Each match is a dictionary containing 'score' and 'metadata' keys. The 'metadata' dictionary must contain 'content' and 'role' keys.

        top_k : int, optional
            The number of top matches to return. Default is 5.

        threshold : float, optional
            The score threshold below which matches will be pruned. Default is 0.91.

        role_weight : float, optional
            The weight to apply if the role of a match is 'assistant'. Default is 1.2.

        length_weight : float, optional
            The weight to apply if the match's content length is greater than the median length. Default is 1.2.

        semantic_bias : float, optional
            A constant bias to add to each match's score before applying weights. Default is 0.

        Returns
        -------
        list[dict]
            The top k matches after rescoring, pruning, and sorting. Each match is a dictionary containing at least 'score' and 'metadata' keys.
        """

        median_length = sorted([len(match['metadata']['content'])
                               for match in matches])[len(matches) // 2]

        for match in matches:
            original_score = match['score']
            role_weight = 1.2 if match['metadata']['role'] == 'assistant' else 1
            length_weight = 1.2 if len(
                match['metadata']['content']) > median_length else 0.9

            match['score'] = (match['score'] + semantic_bias) * \
                ((role_weight + length_weight) / 2)

            score_diff = match['score'] - original_score
            if score_diff > 0.2:
                match['score'] = original_score

            if self.debug:
                # print(f"Role: {match['metadata']['role']}, Length: {len(match['metadata']['content'])}, Score: {round(match['score'], 5)} (Original: {round(original_score, 5)})")
                logging.info("Role: %s, Length: %s, Score: %s (Original: %s, Diff: %s)", match['metadata']['role'], len(
                    match['metadata']['content']), round(match['score'], 5), round(original_score, 5), round(score_diff, 5))

        # prune matches
        matches = [match for match in matches if match['score'] > threshold]

        # Sort the matches based on the new score
        matches.sort(key=lambda x: x['score'], reverse=True)

        # Return the top k matches
        matches = matches[:top_k]

        return matches

    def query_index(self, query=None, top_k=5, threshold=0.9, role_weight=1.2, length_weight=1.2, semantic_bias=0, rescoring=True, silent=False):
        """
        Queries the instance's index with a specified query and optionally rescores the matches.

        This method sets and embeds the query (using an existing query if not provided), then queries the instance's index for matches.
        The matches are initially sorted by score, and then optionally rescored according to custom criteria (role, length, and semantic bias).
        The top k matches are stored in the instance's context attribute and returned.

        Parameters
        ----------
        query : str, optional
            The query to use. If not provided, uses the instance's current query if it exists. If no current query exists, raises a ValueError.

        top_k : int, optional
            The number of top matches to return and store in context. Default is 5.

        threshold : float, optional
            The score threshold used for rescoring. Default is 0.9.

        role_weight : float, optional
            The weight to apply to 'assistant' role matches during rescoring. Default is 1.2.

        length_weight : float, optional
            The weight to apply to matches with content length greater than median during rescoring. Default is 1.2.

        semantic_bias : float, optional
            A constant bias added to each match's score during rescoring. Default is 0.

        rescoring : bool, optional
            A flag indicating whether to perform rescoring. Default is True.

        Returns
        -------
        list[dict]
            The top k matches after querying and optional rescoring. Each match is a dictionary containing at least 'score' and 'metadata' keys.

        Raises
        ------
        ValueError
            If both the provided query and the instance's current query are empty.
        """

        if query is None and self.query != '':
            self.__embed_query()
        elif query is None and self.query == '':
            raise ValueError("Query is empty")
        else:
            self.query = query
            self.__embed_query()

        query_result = self.index.query(
            vector=self.query_embed, top_k=20, include_metadata=True, namespace=self.user_id)
        matches = query_result.to_dict()['matches']
        # sort matches by score
        matches.sort(key=lambda x: x['score'], reverse=True)

        if rescoring:
            self.context = self.__rescore_matches(matches, top_k=top_k, threshold=threshold,
                                                  role_weight=role_weight, length_weight=length_weight, semantic_bias=semantic_bias)
        else:
            self.context = matches[:top_k]

        if not silent:
            return self.context

    def generate_prompt(self, preface_message=DEFAULT_PREFACE):
        """
        Generates prompt for GPT-4 response based on the instance's context.
        """

        if not self.context:
            raise ValueError("Context is empty")

        if not self.query:
            raise ValueError("Query is empty")

        preface = [{'role': 'system', 'content': preface_message}]
        context = [{'role': match['metadata']['role'], 'content': match['metadata']['content']} for match in self.context]
        query = [{'role': 'user', 'content': self.query}]
        self.prompt = preface + context + query

        return self.prompt

    def generate_response(self, query=None, save_to_context=False):
        """
        Generates a response to the instance's current query and optionally saves the response to the instance's context.

        This method first checks if a prompt has already been generated. If not, it generates a prompt using the instance's 
        context and query. Then, it uses the generated prompt to get a response from the GPT-4 model.

        If save_to_context flag is set to True, this method also updates the context by removing the match with the lowest 
        similarity score and adding the new response text as an 'assistant' role match.

        Parameters
        ----------
        save_to_context : bool, optional
            A flag indicating whether to save the generated response to the instance's context. Default is False.

        Returns
        -------
        dict
            The generated response dictionary, as returned by the GPT-4 model's 'ChatCompletion.create' method.
        """

        if self.debug:
            logging.debug("Prompt is empty, generating prompt")
        if query:
            self.set_query(query)
        self.generate_prompt()

        response = openai.ChatCompletion.create(model="gpt-4", messages=self.prompt)
        self.response = response

        if save_to_context:
            # remove the context entry with the lowest score and append the new response text
            scores_and_ids = [(match['score'], match['id']) for match in self.context]
            min_score_id = min(scores_and_ids, key=lambda x: x[0])[1]
            self.context.remove(next((match for match in self.context if match['id'] == min_score_id)))
            max_score = max([match['score'] for match in self.context])

            self.context.append(
                {'id': uuid.uuid4(), 'metadata': {'role': 'assistant', 'content': self.response['choices'][0]['message']['content']}, 'score': (max_score + .1) }) # type: ignore

        return response
