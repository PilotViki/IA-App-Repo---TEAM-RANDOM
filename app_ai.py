import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import warnings
import os
import threading
warnings.filterwarnings('ignore')

# Setup matplotlib
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['text.color'] = 'black'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class DataLoader:
    """Load datasets from text files"""
    
    @staticmethod
    def check_datasets():
        """Check if dataset files exist"""
        datasets_dir = 'datasets'
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            return False
        
        required_files = ['amazon_reviews.txt', 'news_articles.txt', 'customer_reviews.txt']
        for file in required_files:
            if not os.path.exists(os.path.join(datasets_dir, file)):
                return False
        return True
    
    @staticmethod
    def load_dataset(dataset_name):
        """Load dataset from text file"""
        file_path = f'datasets/{dataset_name}.txt'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        texts = []
        labels = []
        category_mapping = {}
        current_category_id = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse format: CATEGORY || TEXT
                if '||' in line:
                    parts = line.split('||', 1)
                    category = parts[0].strip()
                    text = parts[1].strip()
                    
                    # Create mapping for categories
                    if category not in category_mapping:
                        category_mapping[category] = current_category_id
                        current_category_id += 1
                    
                    texts.append(text)
                    labels.append(category_mapping[category])
        
        # Get category names in order
        category_names = [None] * len(category_mapping)
        for cat, idx in category_mapping.items():
            category_names[idx] = cat
        
        return texts, labels, category_names


class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.distances = self._compute_distances()
        self.bkt_nodes_explored = 0
        self.bkt_best_length = float('inf')
        self.bkt_best_path = None
    
    def _compute_distances(self):
        dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dist[i][j] = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                                         (self.cities[i][1] - self.cities[j][1])**2)
        return dist
    
    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distances[path[i]][path[i+1]]
        length += self.distances[path[-1]][path[0]]
        return length
    
    def branch_and_bound(self, timeout_seconds=30):
        """Branch and Bound algorithm for TSP with timeout"""
        self.bkt_nodes_explored = 0
        self.bkt_best_length = float('inf')
        self.bkt_best_path = None
        
        start_time = time.time()
        
        def lower_bound(path):
            if len(path) == self.n:
                return self.path_length(path)
            
            current_sum = 0
            for i in range(len(path) - 1):
                current_sum += self.distances[path[i]][path[i+1]]
            
            unvisited = set(range(self.n)) - set(path)
            bound = current_sum
            
            for node in unvisited:
                min_edge = min(self.distances[node][j] for j in range(self.n) if j != node)
                bound += min_edge
            
            if path:
                last = path[-1]
                if unvisited:
                    bound += min(self.distances[last][j] for j in unvisited)
            
            if path and unvisited:
                first = path[0]
                bound += min(self.distances[first][j] for j in unvisited)
            
            return bound
        
        def dfs(path, visited):
            nonlocal start_time
            
            if time.time() - start_time > timeout_seconds:
                return
            
            self.bkt_nodes_explored += 1
            
            if len(path) == self.n:
                length = self.path_length(path)
                if length < self.bkt_best_length:
                    self.bkt_best_length = length
                    self.bkt_best_path = path.copy()
                return
            
            lb = lower_bound(path)
            if lb >= self.bkt_best_length:
                return
            
            last = path[-1] if path else 0
            candidates = []
            
            for city in range(self.n):
                if city not in visited:
                    priority = self.distances[last][city]
                    candidates.append((priority, city))
            
            candidates.sort()
            
            for _, city in candidates:
                path.append(city)
                visited.add(city)
                dfs(path, visited)
                path.pop()
                visited.remove(city)
        
        initial_path = [0]
        initial_visited = {0}
        dfs(initial_path, initial_visited)
        
        exec_time = time.time() - start_time
        return self.bkt_best_path, self.bkt_best_length, exec_time, self.bkt_nodes_explored
    
    def nearest_neighbor(self, start_city=0):
        unvisited = set(range(self.n))
        current = start_city
        path = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distances[current][city])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return path, self.path_length(path)
    
    def hill_climbing(self, max_iter=500):
        current_path, current_length = self.nearest_neighbor()
        
        for _ in range(max_iter):
            improved = False
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_path = current_path.copy()
                    new_path[i:j+1] = reversed(new_path[i:j+1])
                    new_length = self.path_length(new_path)
                    
                    if new_length < current_length:
                        current_path = new_path
                        current_length = new_length
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_path, current_length
    
    def simulated_annealing(self, initial_temp=100, cooling_rate=0.995, max_iter=3000):
        current_path, current_length = self.nearest_neighbor()
        best_path = current_path.copy()
        best_length = current_length
        temp = initial_temp
        history = []
        
        for i in range(max_iter):
            idx1, idx2 = random.sample(range(self.n), 2)
            new_path = current_path.copy()
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
            new_length = self.path_length(new_path)
            
            if new_length < current_length:
                current_path = new_path
                current_length = new_length
            else:
                delta = new_length - current_length
                if random.random() < math.exp(-delta / temp):
                    current_path = new_path
                    current_length = new_length
            
            if current_length < best_length:
                best_path = current_path.copy()
                best_length = current_length
            
            temp *= cooling_rate
            
            if i % (max_iter // 20) == 0:
                history.append(best_length)
        
        return best_path, best_length, history
    
    def genetic_algorithm(self, pop_size=50, generations=200, mutation_rate=0.1):
        actual_pop_size = min(pop_size, self.n * 3)
        
        def create_individual():
            return random.sample(range(self.n), self.n)
        
        def crossover(parent1, parent2):
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))
            child = [-1] * size
            child[start:end] = parent1[start:end]
            
            pos = 0
            for i in range(size):
                if child[i] == -1:
                    while parent2[pos] in child:
                        pos += 1
                    child[i] = parent2[pos]
                    pos += 1
            return child
        
        def mutate(individual):
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(self.n), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            return individual
        
        def fitness(path):
            return 1.0 / (self.path_length(path) + 1e-10)
        
        population = []
        attempts = 0
        while len(population) < actual_pop_size and attempts < 2000:
            ind = create_individual()
            if ind not in population:
                population.append(ind)
            attempts += 1
        
        best_path = None
        best_length = float('inf')
        history = []
        
        for generation in range(generations):
            fitnesses = [fitness(ind) for ind in population]
            best_idx = np.argmax(fitnesses)
            current_best = population[best_idx]
            current_length = self.path_length(current_best)
            
            if current_length < best_length:
                best_path = current_best.copy()
                best_length = current_length
            
            if generation % (generations // 20) == 0:
                history.append(best_length)
            
            new_population = [best_path.copy()]
            
            while len(new_population) < actual_pop_size:
                tournament_size = 3
                participants = random.sample(list(zip(population, fitnesses)), tournament_size)
                parent1 = max(participants, key=lambda x: x[1])[0]
                
                participants = random.sample(list(zip(population, fitnesses)), tournament_size)
                parent2 = max(participants, key=lambda x: x[1])[0]
                
                child = crossover(parent1.copy(), parent2.copy())
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_path, best_length, history
    
    def plot_path(self, path, title, figure):
        figure.clear()
        ax = figure.add_subplot(111)
        figure.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        cities_array = np.array(self.cities)
        path_cities = cities_array[path + [path[0]]]
        
        ax.plot(path_cities[:, 0], path_cities[:, 1], color='black', linestyle='-', linewidth=2, 
                marker='o', markersize=8, markerfacecolor='black', markeredgecolor='black')
        ax.plot(cities_array[:, 0], cities_array[:, 1], 'wo', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5)
        
        for i, (x, y) in enumerate(self.cities):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='black')
        
        ax.set_title(title, fontsize=12, fontweight='bold', color='black')
        ax.set_xlabel("X Coordinate", fontsize=10, color='black')
        ax.set_ylabel("Y Coordinate", fontsize=10, color='black')
        ax.grid(True, alpha=0.2, color='black')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        for spine in ax.spines.values():
            spine.set_color('black')
        
        figure.tight_layout()
        return figure


class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', 
                                          lowercase=True, ngram_range=(1, 2))
        self.classifiers = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'SVM': SVC(kernel='linear', random_state=42, C=1.0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.trained_models = {}
        self.current_dataset = None
        self.category_names = []
    
    def load_dataset(self, dataset_name, progress_callback=None):
        if progress_callback:
            progress_callback(f"Loading {dataset_name} dataset...", 5)
        
        texts, labels, category_names = DataLoader.load_dataset(dataset_name)
        self.category_names = category_names
        
        if progress_callback:
            progress_callback(f"Loaded {len(texts)} examples", 10)
        
        return texts, labels
    
    def train(self, dataset_name='amazon_reviews', progress_callback=None):
        texts, labels = self.load_dataset(dataset_name, progress_callback)
        self.current_dataset = dataset_name
        y = np.array(labels)
        
        if progress_callback:
            progress_callback("Vectorizing text with TF-IDF...", 20)
        
        X = self.vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        if progress_callback:
            progress_callback("Training classifiers...", 30)
        
        results = {}
        total_classifiers = len(self.classifiers)
        
        for idx, (name, clf) in enumerate(self.classifiers.items()):
            if progress_callback:
                progress_callback(f"Training {name}...", 30 + int((idx / total_classifiers) * 60))
            
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = 0
                cv_std = 0
            
            results[name] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'classifier': clf,
                'predictions': y_pred,
                'y_test': y_test
            }
        
        self.trained_models = results
        
        if progress_callback:
            progress_callback("Training complete", 100)
        
        return results
    
    def predict_category(self, text):
        if not self.trained_models:
            return None
        
        vectorized = self.vectorizer.transform([text.lower()])
        predictions = {}
        
        for name, model_info in self.trained_models.items():
            clf = model_info['classifier']
            pred = clf.predict(vectorized)[0]
            predictions[name] = self.category_names[pred]
        
        return predictions
    
    def plot_comparison(self, results, figure):
        figure.clear()
        
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        figure.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        names = list(results.keys())
        accuracies = [results[n]['accuracy'] for n in names]
        times = [results[n]['train_time'] for n in names]
        
        bars1 = ax1.bar(names, accuracies, color='black', alpha=0.7)
        ax1.set_ylabel('Accuracy', fontsize=10, color='black')
        ax1.set_title('Classifier Accuracy Comparison', fontsize=11, fontweight='bold', color='black')
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.2, axis='y')
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
        
        bars2 = ax2.bar(names, times, color='gray', alpha=0.7)
        ax2.set_ylabel('Time (seconds)', fontsize=10, color='black')
        ax2.set_title('Training Time Comparison', fontsize=11, fontweight='bold', color='black')
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.2, axis='y')
        
        for bar, time_val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.05,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8, color='black')
        
        for spine in ax1.spines.values():
            spine.set_color('black')
        for spine in ax2.spines.values():
            spine.set_color('black')
        
        figure.tight_layout()
        return figure


class AIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Proiect IA - Laborator | ECHIPA: RANDOM | TSP + BKT + NLP")
        self.root.geometry("1400x850")
        self.root.configure(bg='#f0f0f0')
        
        self.header_font = font.Font(family='Helvetica', size=11, weight='bold')
        self.body_font = font.Font(family='Helvetica', size=10)
        self.code_font = font.Font(family='Menlo', size=9)
        
        self.create_header()
        
        # Check if datasets exist
        if not DataLoader.check_datasets():
            messagebox.showwarning("Missing Datasets", 
                "Dataset files not found. Please create the 'datasets' folder and add the text files:\n"
                "- amazon_reviews.txt\n"
                "- news_articles.txt\n"
                "- customer_reviews.txt\n\n"
                "Each file should have format: CATEGORY || TEXT")
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.tsp_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.nlp_tab = tk.Frame(self.notebook, bg='#ffffff')
        
        self.notebook.add(self.tsp_tab, text="Travelling Salesman Problem - TSP")
        self.notebook.add(self.nlp_tab, text="Text Classification - NLP")
        
        self.init_tsp_tab()
        self.init_nlp_tab()
        
        self.tsp = None
    
    def create_header(self):
        header = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="MAIN MENU - PROIECT LABORATOR IA", 
                               font=('Helvetica', 16, 'bold'), bg='#1a1a1a', fg='white')
        title_label.place(x=20, y=28)
        
        team_label = tk.Label(header, text="ECHIPA: RANDOM | TSP + BKT + NLP", 
                              font=('Helvetica', 13, 'bold'), bg='#1a1a1a', fg='white')
        team_label.place(relx=1.0, x=-20, y=28, anchor='ne')
    
    def init_tsp_tab(self):
        main = tk.Frame(self.tsp_tab, bg='#ffffff')
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        left = tk.Frame(main, bg='#f5f5f5', width=450, relief='solid', bd=1)
        left.pack(side='left', fill='y', padx=(0, 15))
        left.pack_propagate(False)
        
        right = tk.Frame(main, bg='#ffffff')
        right.pack(side='right', fill='both', expand=True)
        
        self.tsp_figure = Figure(figsize=(7, 6), facecolor='white', dpi=100)
        self.tsp_canvas = FigureCanvasTkAgg(self.tsp_figure, right)
        self.tsp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        config_frame = tk.LabelFrame(left, text="Configuration", font=self.header_font,
                                      bg='#f5f5f5', fg='black', relief='solid', bd=1)
        config_frame.pack(fill='x', padx=15, pady=15)
        
        city_frame = tk.Frame(config_frame, bg='#f5f5f5')
        city_frame.pack(fill='x', padx=15, pady=12)
        tk.Label(city_frame, text="Number of Cities (5-20):", font=self.body_font, 
                bg='#f5f5f5', fg='black').pack(side='left')
        self.city_count = tk.IntVar(value=10)
        spinner = tk.Spinbox(city_frame, from_=5, to=20, textvariable=self.city_count,
                            width=10, font=self.body_font, bg='white', fg='black')
        spinner.pack(side='right')
        
        self.gen_btn = tk.Button(config_frame, text="Generate Random Cities", command=self.generate_cities,
                                 font=self.body_font, bg='black', fg='white',
                                 relief='flat', padx=15, pady=8)
        self.gen_btn.pack(fill='x', padx=15, pady=(0, 12))
        
        algo_frame = tk.LabelFrame(left, text="Algorithm Selection", font=self.header_font,
                                    bg='#f5f5f5', fg='black', relief='solid', bd=1)
        algo_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.algorithm_var = tk.StringVar(value="NN")
        algorithms = [
            ("Nearest Neighbor (NN)", "NN"),
            ("Hill Climbing (HC)", "HC"),
            ("Simulated Annealing (SA)", "SA"),
            ("Genetic Algorithm (GA)", "GA"),
            ("Branch & Bound (BKT)", "BKT"),
            ("Compare All Methods", "ALL")
        ]
        
        for text, value in algorithms:
            rb = tk.Radiobutton(algo_frame, text=text, variable=self.algorithm_var, value=value,
                                font=self.body_font, bg='#f5f5f5', fg='black', selectcolor='#f5f5f5',
                                activebackground='#f5f5f5', activeforeground='black')
            rb.pack(anchor='w', padx=15, pady=3)
        
        params_frame = tk.LabelFrame(left, text="Algorithm Parameters", font=self.header_font,
                                      bg='#f5f5f5', fg='black', relief='solid', bd=1)
        params_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        sa_frame = tk.Frame(params_frame, bg='#f5f5f5')
        sa_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(sa_frame, text="SA Initial Temp:", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.sa_temp = tk.IntVar(value=100)
        tk.Spinbox(sa_frame, from_=50, to=500, textvariable=self.sa_temp, width=8).pack(side='right')
        
        sa_cool_frame = tk.Frame(params_frame, bg='#f5f5f5')
        sa_cool_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(sa_cool_frame, text="SA Cooling Rate:", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.sa_cooling = tk.DoubleVar(value=0.995)
        tk.Spinbox(sa_cool_frame, from_=0.9, to=0.999, increment=0.001, textvariable=self.sa_cooling, width=8).pack(side='right')
        
        ga_frame = tk.Frame(params_frame, bg='#f5f5f5')
        ga_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(ga_frame, text="GA Population:", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.ga_pop = tk.IntVar(value=50)
        tk.Spinbox(ga_frame, from_=20, to=200, textvariable=self.ga_pop, width=8).pack(side='right')
        
        ga_gen_frame = tk.Frame(params_frame, bg='#f5f5f5')
        ga_gen_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(ga_gen_frame, text="GA Generations:", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.ga_gen = tk.IntVar(value=150)
        tk.Spinbox(ga_gen_frame, from_=50, to=500, textvariable=self.ga_gen, width=8).pack(side='right')
        
        ga_mut_frame = tk.Frame(params_frame, bg='#f5f5f5')
        ga_mut_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(ga_mut_frame, text="GA Mutation Rate:", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.ga_mut = tk.DoubleVar(value=0.1)
        tk.Spinbox(ga_mut_frame, from_=0.01, to=0.5, increment=0.01, textvariable=self.ga_mut, width=8).pack(side='right')
        
        bkt_frame = tk.Frame(params_frame, bg='#f5f5f5')
        bkt_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(bkt_frame, text="BKT Timeout (sec):", font=self.body_font, bg='#f5f5f5', fg='black').pack(side='left')
        self.bkt_timeout = tk.IntVar(value=30)
        tk.Spinbox(bkt_frame, from_=5, to=60, textvariable=self.bkt_timeout, width=8).pack(side='right')
        
        self.run_btn = tk.Button(left, text="Run Algorithm", command=self.run_tsp,
                                 font=('Helvetica', 12, 'bold'), bg='black', fg='white',
                                 relief='flat', padx=15, pady=10)
        self.run_btn.pack(fill='x', padx=15, pady=(0, 15))
        
        results_frame = tk.LabelFrame(left, text="Results", font=self.header_font,
                                       bg='#f5f5f5', fg='black', relief='solid', bd=1)
        results_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.tsp_results = scrolledtext.ScrolledText(results_frame, height=12,
                                                      font=self.code_font, bg='white', fg='black')
        self.tsp_results.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.generate_cities()
    
    def generate_cities(self):
        np.random.seed(42)
        n = self.city_count.get()
        cities = [(np.random.uniform(10, 90), np.random.uniform(10, 90)) for _ in range(n)]
        self.tsp = TSP(cities)
        
        self.tsp_figure.clear()
        ax = self.tsp_figure.add_subplot(111)
        ax.set_facecolor('white')
        cities_arr = np.array(cities)
        ax.plot(cities_arr[:, 0], cities_arr[:, 1], 'ko', markersize=8, 
                markeredgecolor='black', markerfacecolor='white')
        
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, fontweight='bold', color='black')
        
        ax.set_title(f"City Locations ({n} cities)", fontsize=12, fontweight='bold', color='black')
        ax.set_xlabel("X Coordinate", fontsize=10, color='black')
        ax.set_ylabel("Y Coordinate", fontsize=10, color='black')
        ax.grid(True, alpha=0.2, color='black')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        for spine in ax.spines.values():
            spine.set_color('black')
        
        self.tsp_canvas.draw()
        self.tsp_results.insert(tk.END, f"Generated {n} cities\n")
        self.tsp_results.see(tk.END)
    
    def run_tsp(self):
        if self.tsp is None:
            messagebox.showerror("Error", "Generate cities first")
            return
        
        algorithm = self.algorithm_var.get()
        self.run_btn.config(state='disabled', text='Running...')
        self.tsp_results.insert(tk.END, f"\n{'='*60}\n")
        self.tsp_results.insert(tk.END, f"Running {algorithm}...\n")
        self.tsp_results.update()
        
        start_time = time.time()
        
        try:
            if algorithm == "NN":
                path, length = self.tsp.nearest_neighbor()
                self.show_tsp_result("Nearest Neighbor", path, length, time.time() - start_time)
                self.tsp.plot_path(path, f"Nearest Neighbor | Length: {length:.2f}", self.tsp_figure)
                self.tsp_canvas.draw()
            
            elif algorithm == "HC":
                path, length = self.tsp.hill_climbing(max_iter=500)
                self.show_tsp_result("Hill Climbing", path, length, time.time() - start_time)
                self.tsp.plot_path(path, f"Hill Climbing | Length: {length:.2f}", self.tsp_figure)
                self.tsp_canvas.draw()
            
            elif algorithm == "SA":
                path, length, history = self.tsp.simulated_annealing(
                    initial_temp=self.sa_temp.get(),
                    cooling_rate=self.sa_cooling.get(),
                    max_iter=3000
                )
                self.show_tsp_result("Simulated Annealing", path, length, time.time() - start_time)
                self.tsp.plot_path(path, f"Simulated Annealing | Length: {length:.2f}", self.tsp_figure)
                self.tsp_canvas.draw()
                
                if history:
                    self.plot_convergence_window(history, "Simulated Annealing")
            
            elif algorithm == "GA":
                path, length, history = self.tsp.genetic_algorithm(
                    pop_size=self.ga_pop.get(),
                    generations=self.ga_gen.get(),
                    mutation_rate=self.ga_mut.get()
                )
                self.show_tsp_result("Genetic Algorithm", path, length, time.time() - start_time)
                self.tsp.plot_path(path, f"Genetic Algorithm | Length: {length:.2f}", self.tsp_figure)
                self.tsp_canvas.draw()
                
                if history:
                    self.plot_convergence_window(history, "Genetic Algorithm")
            
            elif algorithm == "BKT":
                path, length, exec_time, nodes = self.tsp.branch_and_bound(timeout_seconds=self.bkt_timeout.get())
                self.show_tsp_result("Branch & Bound (BKT)", path, length, exec_time, 
                                     additional_info=f"Nodes explored: {nodes}")
                if path:
                    self.tsp.plot_path(path, f"BKT | Length: {length:.2f} | Nodes: {nodes}", self.tsp_figure)
                    self.tsp_canvas.draw()
                else:
                    self.tsp_results.insert(tk.END, "BKT did not find a solution within timeout\n")
            
            elif algorithm == "ALL":
                self.compare_all_methods()
        
        except Exception as e:
            self.tsp_results.insert(tk.END, f"ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
        
        finally:
            self.run_btn.config(state='normal', text='Run Algorithm')
            self.tsp_results.see(tk.END)
    
    def show_tsp_result(self, name, path, length, exec_time, additional_info=""):
        self.tsp_results.insert(tk.END, f"\n{name}\n")
        self.tsp_results.insert(tk.END, f"   Path Length: {length:.2f}\n")
        self.tsp_results.insert(tk.END, f"   Execution Time: {exec_time:.3f} seconds\n")
        if additional_info:
            self.tsp_results.insert(tk.END, f"   {additional_info}\n")
        self.tsp_results.insert(tk.END, f"   Path: {path[:10]}... (showing first 10 cities)\n")
        self.tsp_results.insert(tk.END, f"{'-'*60}\n")
    
    def plot_convergence_window(self, history, title):
        fig = Figure(figsize=(8, 5), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.plot(history, 'k-', linewidth=2)
        ax.set_xlabel("Iteration", fontsize=11, color='black')
        ax.set_ylabel("Best Path Length", fontsize=11, color='black')
        ax.set_title(f"Convergence - {title}", fontsize=12, fontweight='bold', color='black')
        ax.grid(True, alpha=0.3, color='black')
        for spine in ax.spines.values():
            spine.set_color('black')
        fig.tight_layout()
        fig.show()
    
    def compare_all_methods(self):
        methods = [
            ("NN", lambda: (self.tsp.nearest_neighbor()[0], self.tsp.nearest_neighbor()[1])),
            ("HC", lambda: (self.tsp.hill_climbing(max_iter=500)[0], self.tsp.hill_climbing(max_iter=500)[1])),
            ("SA", lambda: (self.tsp.simulated_annealing(max_iter=2000)[0], 
                           self.tsp.simulated_annealing(max_iter=2000)[1])),
            ("GA", lambda: (self.tsp.genetic_algorithm(generations=100)[0], 
                           self.tsp.genetic_algorithm(generations=100)[1])),
            ("BKT", lambda: (self.tsp.branch_and_bound(timeout_seconds=20)[0], 
                            self.tsp.branch_and_bound(timeout_seconds=20)[1]))
        ]
        
        results = []
        self.tsp_results.insert(tk.END, "\nCOMPARING ALL METHODS\n")
        self.tsp_results.insert(tk.END, "="*60 + "\n")
        
        for name, method in methods:
            start = time.time()
            try:
                path, length = method()
                exec_time = time.time() - start
                if path:
                    results.append((name, length, exec_time))
                    self.tsp_results.insert(tk.END, f"  {name:4} : Length={length:8.2f}  Time={exec_time:.3f}s\n")
                else:
                    self.tsp_results.insert(tk.END, f"  {name:4} : No solution found\n")
                self.tsp_results.update()
            except Exception as e:
                self.tsp_results.insert(tk.END, f"  {name:4} : ERROR - {str(e)[:30]}\n")
        
        if results:
            self.plot_method_comparison(results)
    
    def plot_method_comparison(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        names = [r[0] for r in results]
        lengths = [r[1] for r in results]
        times = [r[2] for r in results]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars1 = ax1.bar(names, lengths, color=colors[:len(names)], alpha=0.7)
        ax1.set_ylabel("Path Length", fontsize=11, color='black')
        ax1.set_title("Solution Quality Comparison", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2, axis='y')
        
        for bar, length in zip(bars1, lengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lengths)*0.01,
                    f'{length:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        bars2 = ax2.bar(names, times, color=colors[:len(names)], alpha=0.7)
        ax2.set_ylabel("Time (seconds)", fontsize=11, color='black')
        ax2.set_title("Execution Time Comparison", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='y')
        
        for bar, time_val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.05,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            for spine in ax.spines.values():
                spine.set_color('black')
        
        plt.tight_layout()
        plt.show()
    
    def init_nlp_tab(self):
        main = tk.Frame(self.nlp_tab, bg='#ffffff')
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        left = tk.Frame(main, bg='#f5f5f5', width=450, relief='solid', bd=1)
        left.pack(side='left', fill='y', padx=(0, 15))
        left.pack_propagate(False)
        
        right = tk.Frame(main, bg='#ffffff')
        right.pack(side='right', fill='both', expand=True)
        
        self.nlp_notebook = ttk.Notebook(right)
        self.nlp_notebook.pack(fill='both', expand=True)
        
        self.acc_frame = tk.Frame(self.nlp_notebook, bg='#ffffff')
        self.nlp_notebook.add(self.acc_frame, text="Accuracy Comparison")
        self.nlp_figure = Figure(figsize=(7, 6), facecolor='white', dpi=100)
        self.nlp_canvas = FigureCanvasTkAgg(self.nlp_figure, self.acc_frame)
        self.nlp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.info_frame = tk.Frame(self.nlp_notebook, bg='#ffffff')
        self.nlp_notebook.add(self.info_frame, text="Dataset Info")
        self.info_text = scrolledtext.ScrolledText(self.info_frame, font=self.code_font, 
                                                     bg='white', fg='black')
        self.info_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        dataset_frame = tk.LabelFrame(left, text="Dataset Selection", font=self.header_font,
                                       bg='#f5f5f5', fg='black', relief='solid', bd=1)
        dataset_frame.pack(fill='x', padx=15, pady=15)
        
        self.dataset_var = tk.StringVar(value="amazon_reviews")
        datasets = [
            ("Amazon Product Reviews", "amazon_reviews"),
            ("News Articles", "news_articles"),
            ("Customer Reviews", "customer_reviews")
        ]
        
        for text, value in datasets:
            rb = tk.Radiobutton(dataset_frame, text=text, variable=self.dataset_var, value=value,
                                font=self.body_font, bg='#f5f5f5', fg='black', selectcolor='#f5f5f5',
                                activebackground='#f5f5f5', activeforeground='black')
            rb.pack(anchor='w', padx=15, pady=5)
        
        self.train_btn = tk.Button(dataset_frame, text="Train Classifiers", command=self.train_models,
                                   font=('Helvetica', 11, 'bold'), bg='black', fg='white',
                                   relief='flat', padx=15, pady=10)
        self.train_btn.pack(fill='x', padx=15, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(dataset_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=15, pady=5)
        
        self.status_label = tk.Label(dataset_frame, text="Select dataset and train", font=self.body_font,
                                      bg='#f5f5f5', fg='black')
        self.status_label.pack(pady=(0, 10))
        
        pred_frame = tk.LabelFrame(left, text="Text Classification", font=self.header_font,
                                    bg='#f5f5f5', fg='black', relief='solid', bd=1)
        pred_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        tk.Label(pred_frame, text="Enter text to classify:", font=self.body_font,
                bg='#f5f5f5', fg='black').pack(anchor='w', padx=15, pady=(10, 5))
        
        self.text_input = scrolledtext.ScrolledText(pred_frame, height=5, font=self.body_font,
                                                     bg='white', fg='black', relief='solid', bd=1)
        self.text_input.pack(fill='x', padx=15, pady=5)
        
        self.classify_btn = tk.Button(pred_frame, text="Classify Text", command=self.classify_text,
                                      font=self.body_font, bg='black', fg='white',
                                      relief='flat', padx=15, pady=8)
        self.classify_btn.pack(fill='x', padx=15, pady=10)
        
        res_frame = tk.LabelFrame(left, text="Classification Results", font=self.header_font,
                                   bg='#f5f5f5', fg='black', relief='solid', bd=1)
        res_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.nlp_results = scrolledtext.ScrolledText(res_frame, height=10,
                                                      font=self.code_font, bg='white', fg='black')
        self.nlp_results.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.classifier = TextClassifier()
        self.is_trained = False
        
        self.load_dataset_info()
    
    def load_dataset_info(self):
        info_text = "AVAILABLE DATASETS\n\n"
        info_text += "="*50 + "\n\n"
        
        datasets = [
            ("amazon_reviews", "Amazon Product Reviews", 40, "Electronics, Books, Movies, Clothing"),
            ("news_articles", "News Articles", 40, "Politics, Business, Sports, Technology"),
            ("customer_reviews", "Customer Reviews", 40, "Restaurants, Hotels, Airlines, Products")
        ]
        
        for name, title, count, categories in datasets:
            info_text += f"Dataset: {title}\n"
            info_text += f"   File: {name}.txt\n"
            info_text += f"   Examples: {count}\n"
            info_text += f"   Categories: {categories}\n\n"
        
        info_text += "="*50 + "\n"
        info_text += "TIP: You can edit the text files in the 'datasets' folder\n"
        info_text += "to add your own examples or create new datasets.\n"
        info_text += "Format: CATEGORY || TEXT\n"
        
        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state='disabled')
    
    def update_progress(self, msg, val):
        self.progress_var.set(val)
        self.status_label.config(text=msg, fg='black')
        self.root.update_idletasks()
    
    def train_models(self):
        self.train_btn.config(state='disabled', text='Training...')
        self.classify_btn.config(state='disabled')
        dataset = self.dataset_var.get()
        self.nlp_results.insert(tk.END, f"\n{'='*60}\n")
        self.nlp_results.insert(tk.END, f"Training on dataset: {dataset}\n")
        self.nlp_results.update()
        
        def thread_func():
            try:
                results = self.classifier.train(dataset_name=dataset, progress_callback=self.update_progress)
                self.is_trained = True
                self.root.after(0, self.training_done, results)
            except Exception as e:
                self.root.after(0, self.training_error, str(e))
        
        threading.Thread(target=thread_func, daemon=True).start()
    
    def training_done(self, results):
        self.train_btn.config(state='normal', text='Train Classifiers')
        self.classify_btn.config(state='normal')
        self.status_label.config(text="Training complete", fg='green')
        
        self.nlp_results.insert(tk.END, "\nTraining Results:\n")
        self.nlp_results.insert(tk.END, "="*60 + "\n")
        self.nlp_results.insert(tk.END, f"{'Classifier':<22} {'Accuracy':<12} {'Time (s)':<10}\n")
        self.nlp_results.insert(tk.END, "-"*60 + "\n")
        
        for name, info in results.items():
            self.nlp_results.insert(tk.END, f"{name:<22} {info['accuracy']:.4f}      {info['train_time']:.3f}\n")
        
        self.nlp_results.insert(tk.END, "\nReady for classification\n")
        self.nlp_results.see(tk.END)
        
        self.classifier.plot_comparison(results, self.nlp_figure)
        self.nlp_canvas.draw()
    
    def training_error(self, error):
        self.train_btn.config(state='normal', text='Train Classifiers')
        self.classify_btn.config(state='normal')
        self.status_label.config(text="Training failed", fg='red')
        self.nlp_results.insert(tk.END, f"ERROR: {error}\n")
        messagebox.showerror("Training Error", f"Training failed:\n{error}")
    
    def classify_text(self):
        if not self.is_trained:
            messagebox.showwarning("Warning", "Please train the classifiers first")
            return
        
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to classify")
            return
        
        self.nlp_results.insert(tk.END, f"\nAnalyzing text:\n")
        self.nlp_results.insert(tk.END, f"   \"{text[:100]}{'...' if len(text) > 100 else ''}\"\n")
        self.nlp_results.insert(tk.END, "-"*60 + "\n")
        
        preds = self.classifier.predict_category(text)
        if preds:
            for name, category in preds.items():
                self.nlp_results.insert(tk.END, f"   {name:<22} -> {category}\n")
        
        self.nlp_results.insert(tk.END, "\n")
        self.nlp_results.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIApplication(root)
    root.mainloop()