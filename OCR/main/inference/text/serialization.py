class SaveAsText:
	def __init__(self, words):
		self.words = words 

	def save(self):
		with open("result.txt", "w") as f:
			for word in self.words:
				f.write(word + "\n")
		f.close()

	def load(self):
		pass