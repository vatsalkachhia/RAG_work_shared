"""
demo.py
-------
Demo CLI for the RAG Engine.
"""

from engine import RAGEngine

if __name__ == "__main__":
    sample_text = (
        """# The Dawn of Human Civilization: A Journey Through Knowledge

## 1. Introduction

Human civilization did not emerge overnight. It was the result of countless generations experimenting, failing, discovering, and innovating. From the invention of fire to the age of Artificial Intelligence, humanity has continuously redefined itself. Understanding this journey not only tells us where we came from but also illuminates where we might be heading.

## 2. The Age of Fire and Stone

The mastery of fire was perhaps the single most important discovery in early human history. It allowed our ancestors to cook food, ward off predators, and migrate to colder climates. Alongside fire came the development of stone tools. Flint knives, spear points, and later polished stone axes provided the foundation for agriculture and hunting.

### 2.1 The Agricultural Revolution

Around 10,000 BCE, humanity began shifting from nomadic lifestyles to settled farming communities. This period, often called the Neolithic Revolution, gave rise to villages and eventually cities. Agriculture allowed food surpluses, which in turn supported population growth, social hierarchies, and specialized professions.

### 2.2 Early Civilizations

The great river valley civilizations—Mesopotamia, the Indus Valley, Ancient Egypt, and the Yellow River in China—developed complex societies. Writing systems such as cuneiform and hieroglyphs emerged to record trade, taxes, and religious rituals. Monumental architecture like pyramids and ziggurats testified to centralized authority and advanced engineering.

## 3. The Rise of Knowledge and Philosophy

As societies stabilized, humans began to ask deeper questions: Who are we? What is the nature of reality? How should we live?

### 3.1 Greek Philosophy

Socrates introduced critical questioning, Plato envisioned ideal forms, and Aristotle classified knowledge systematically. Their ideas laid the groundwork for Western philosophy and science.

### 3.2 Indian Wisdom Traditions

In India, the Vedas and Upanishads explored metaphysics, consciousness, and the universe. Concepts like Dharma, Karma, and Moksha shaped not just religion but social and ethical life. Buddhism, founded by Siddhartha Gautama, spread across Asia with its profound analysis of suffering and liberation.

### 3.3 Chinese Thought

Confucianism emphasized morality, family, and governance, while Daoism focused on harmony with nature and the Dao (the Way). These philosophies continue to shape East Asian societies to this day.

## 4. The Scientific Revolution

The 16th and 17th centuries marked a turning point. Figures like Copernicus, Galileo, and Newton challenged long-held dogmas with mathematics and experimentation. The heliocentric model placed the Sun—not the Earth—at the center of the solar system, forever altering humanity’s place in the cosmos.

### 4.1 Invention of the Printing Press

The printing press, invented by Johannes Gutenberg in the mid-15th century, democratized knowledge. Books could now be mass-produced, spreading ideas rapidly and fueling the Renaissance, Reformation, and Enlightenment.

### 4.2 Enlightenment Values

Thinkers like Voltaire, Rousseau, and Kant emphasized reason, liberty, and equality. These ideals fueled revolutions in America and France, shaping the modern world.

## 5. The Industrial Age

The Industrial Revolution of the 18th and 19th centuries mechanized production, transforming economies and societies. Steam engines, railroads, and factories brought unprecedented progress but also urban poverty, labor exploitation, and environmental challenges.

### 5.1 Science Meets Industry

Electricity, chemistry, and biology advanced rapidly. Charles Darwin’s theory of evolution revolutionized biology, while Faraday and Maxwell’s work on electromagnetism laid the foundation for modern physics and technology.

### 5.2 Social Transformations

Industrialization led to urbanization, mass education, and the rise of nation-states. It also intensified debates about inequality, giving rise to socialist and labor movements.

## 6. The Digital and Information Era

The 20th century introduced computers, the internet, and digital technologies. Information became the new currency of power.

### 6.1 World Wars and Technology

The two World Wars accelerated technological innovation—from radar to nuclear energy. While destructive, they also paved the way for modern computing, aviation, and medicine.

### 6.2 Space Exploration

The launch of Sputnik in 1957 marked the beginning of the space age. Neil Armstrong’s moon landing in 1969 symbolized humanity’s capacity to dream beyond Earth. Today, Mars missions and space telescopes extend our cosmic horizon.

### 6.3 Rise of Artificial Intelligence

From Alan Turing’s early theories to present-day deep learning, AI has become one of humanity’s most transformative technologies. Machine learning, natural language processing, and robotics now influence healthcare, finance, education, and even art.

## 7. Challenges of the 21st Century

While technology has advanced, humanity faces urgent challenges:

* **Climate Change**: Rising temperatures threaten ecosystems and human societies.
* **Inequality**: Wealth and resources remain unevenly distributed.
* **Ethics of AI**: As machines gain more autonomy, questions of fairness, bias, and accountability arise.
* **Global Health**: Pandemics like COVID-19 remind us of the fragility of our interconnected world.

## 8. Conclusion: The Road Ahead

The story of human civilization is far from over. If history teaches us anything, it is that innovation comes with responsibility. The same tools that build can also destroy. Fire once kept us warm but also burned down forests. Nuclear energy powers cities but can annihilate them. Artificial Intelligence may either empower humanity or displace it.

The challenge for the future is not just to create but to create wisely. Civilization is a fragile flame. To carry it forward, humanity must combine knowledge with wisdom, science with ethics, and power with compassion."""
    )

    config = {
        "chunking": "recursive",
        "embedding": "huggingface",
        "vectordb": "faiss",
        "retrieval": "topk",
        "llm": "groq",
        "memory": "windowed",
        "reranker": False,
    }

    rag = RAGEngine(config)
    rag.build_knowledge_base(sample_text)

    print("\n🤖 RAG Prototype Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            result = rag.query(user_input)
            print(f"Bot: {result['answer']}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
