from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SocialMediaMarketing():
	"""SocialMediaMarketing crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def ideator_and_trend_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['ideator_and_trend_analyst'],
			verbose=True
		)

	@agent
	def content_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['content_generator'],
			verbose=True
		)

	@agent
	def social_media_manager(self) -> Agent:
		return Agent(
			config=self.agents_config['social_media_manager'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def ideation_task(self) -> Task:
		return Task(
			config=self.tasks_config['ideation_task'],
		)

	@task
	def content_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['content_creation_task'],
		)

	@task
	def social_media_optimization_task(self) -> Task:
		return Task(
			config=self.tasks_config['social_media_optimization_task'],
			output_file='optimized_content.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SocialMediaMarketing crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
