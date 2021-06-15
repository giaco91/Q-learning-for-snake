from PIL import Image
import pygame,sys,random
from pygame.math import Vector2
import numpy as np
import matplotlib.pyplot as plt

class Graphics:
	#Takes care of rendering the frames
    def __init__(self,cell_size,cell_number,screen):

        self.new_block = False
        self.cell_size=cell_size
        self.cell_number=cell_number
        self.screen=screen
        
        self.game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)
        
        self.snack_image = pygame.image.load('Graphics/mouse_transparent.jpeg').convert_alpha()

        self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
        
        self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
        
        self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')
        
    def draw_elements(self,environment):
        self.draw_grass(environment)
        self.draw_snack(environment.snack_pos)
        self.draw_snake(environment.snake_body)
        self.draw_score(environment.snake_body)

    def draw_snake(self,snake_body):
        self.update_head_graphics(snake_body)
        self.update_tail_graphics(snake_body)
    
        for index,block in enumerate(snake_body):
            x_pos = int((block.x+1) * self.cell_size)
            y_pos = int((block.y+1)* self.cell_size)
            block_rect = pygame.Rect(x_pos,y_pos,self.cell_size,self.cell_size)

            if index == 0:
                self.screen.blit(self.head,block_rect)
            elif index == len(snake_body) - 1:
                self.screen.blit(self.tail,block_rect)
            else:
                previous_block = snake_body[index + 1] - block
                next_block = snake_body[index - 1] - block
                if previous_block.x == next_block.x:
                    self.screen.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    self.screen.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        self.screen.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        self.screen.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        self.screen.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        self.screen.blit(self.body_br,block_rect)

    def update_head_graphics(self,snake_body):
        head_relation = snake_body[1] - snake_body[0]
        if head_relation == Vector2(1,0): self.head = self.head_left
        elif head_relation == Vector2(-1,0): self.head = self.head_right
        elif head_relation == Vector2(0,1): self.head = self.head_up
        elif head_relation == Vector2(0,-1): self.head = self.head_down
            

    def update_tail_graphics(self,snake_body):
        tail_relation = snake_body[-2] - snake_body[-1]
        if tail_relation == Vector2(1,0): self.tail = self.tail_left
        elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
        elif tail_relation == Vector2(0,1): self.tail = self.tail_up
        elif tail_relation == Vector2(0,-1): self.tail = self.tail_down


    def play_crunch_sound(self):
        self.crunch_sound.play()

    def draw_snack(self,snack_pos):
        snack_rect = pygame.Rect(int((snack_pos.x+1)* self.cell_size),int((snack_pos.y+1) * self.cell_size),self.cell_size,self.cell_size)
        self.screen.blit(self.snack_image,snack_rect)


    def draw_grass(self,environment):
        snake_body=environment.snake_body
        game_table=environment.game_table
        map_margin=environment.map_margin
        snack_pos=environment.snack_pos
        head_pos=snake_body[0]

        fac=1.1
        hurdle_color=(90,90,90)
        grass_color_bright = (167,209,61)
        grass_color_bright_bright=(int(fac*grass_color_bright[0]),int(fac*grass_color_bright[1]),int(fac*grass_color_bright[2]))
        grass_color_dark = (180,220,70)
        grass_color_dark_bright=(int(fac*grass_color_dark[0]),int(fac*grass_color_dark[1]),int(fac*grass_color_dark[2]))

        for row in range(self.cell_number+2):
            if row % 2 == 0: 
                for col in range(self.cell_number+2):
                    rect=pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
                    if game_table[col+map_margin-1,row+map_margin-1]==2:
                        color=hurdle_color
                    else:
                        if head_pos[0]-1<=col-1<=head_pos[0]+1 and head_pos[1]-1<=row-1<=head_pos[1]+1:
                            color_bright=grass_color_bright_bright
                            color_dark=grass_color_dark_bright
                        else:
                        	color_bright=grass_color_bright
                        	color_dark=grass_color_dark
                        if col % 2 == 0:
                            #grass_rect = pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
                            color=color_bright
                        else:
                            #grass_rect = pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
                            color=color_dark
                    pygame.draw.rect(self.screen,color,rect) 
                    if color==hurdle_color:
                        pygame.draw.rect(self.screen,(0,0,0),rect,width=4)
            else:
                for col in range(self.cell_number+2):
                    rect=pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
                    if game_table[col+map_margin-1,row+map_margin-1]==2:
                        color=hurdle_color
                        pygame.draw.rect(self.screen,(0,0,0),rect,width=2)
                    else:
                        if head_pos[0]-1<=col-1<=head_pos[0]+1 and head_pos[1]-1<=row-1<=head_pos[1]+1:
                            color_bright=grass_color_bright_bright
                            color_dark=grass_color_dark_bright
                        else:
                            color_bright=grass_color_bright
                            color_dark=grass_color_dark
                        if col % 2 != 0:
                            color=color_bright
                        else:
                            color=color_dark
                    pygame.draw.rect(self.screen,color,rect)
                    if color==hurdle_color:
                        pygame.draw.rect(self.screen,(0,0,0),rect,width=4)
        receptive_field_rect = pygame.Rect((head_pos[0]) * self.cell_size,(head_pos[1]) * self.cell_size,3*self.cell_size,3*self.cell_size)
        pygame.draw.rect(self.screen,(0,0,0),receptive_field_rect,width=2)

        head_pos=self.cell_size*(head_pos+Vector2(1.5,1.5))
        snack_pos=self.cell_size*(snack_pos+Vector2(1.5,1.5))
        pygame.draw.line(self.screen,(255,180,0),(head_pos[0],head_pos[1]),(snack_pos[0],snack_pos[1]),width=2)

    def draw_score(self,snake_body):
        score_text = str(len(snake_body) - 3)
        score_surface = self.game_font.render(score_text,True,(56,74,12))
        score_x = int(self.cell_size * (self.cell_number+2) - 40)
        score_y = int(self.cell_size * (self.cell_number+2) - 20)
        score_rect = score_surface.get_rect(center = (score_x,score_y))
        snack_rect = self.snack_image.get_rect(midright = (score_rect.left,score_rect.centery))
        bg_rect = pygame.Rect(snack_rect.left,snack_rect.top,snack_rect.width + score_rect.width + 6,snack_rect.height)

        pygame.draw.rect(self.screen,(167,209,61),bg_rect)
        self.screen.blit(score_surface,score_rect)
        self.screen.blit(self.snack_image,snack_rect)
        pygame.draw.rect(self.screen,(56,74,12),bg_rect,2)


class Environment:
	#evolves the environment based on the snake action
    def __init__(self,cell_size,cell_number,field_of_view_size=1,random_hurdles=False):
        self.field_of_view_size=field_of_view_size
        self.kernel_size=1+2*field_of_view_size
        self.n_kernel_elements=self.kernel_size**2
        self.map_margin=1+self.field_of_view_size
        self.cell_size=cell_size
        self.cell_number=cell_number
        self.random_hurdles=random_hurdles
        self.reset()

        self.update_game_table()
            
    def turn_left(self):
        self.direction=self.direction.rotate(-90)
    
    def turn_right(self):
        self.direction=self.direction.rotate(90)

    def move_snake(self):
        if self.new_block:
            body_copy = self.snake_body[:]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.snake_body = body_copy[:]
            self.new_block = False
            if len(self.snake_body)==self.cell_number**2:
                self.game_solved=True
        else:
            body_copy = self.snake_body[:-1]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.snake_body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def reset(self):
        self.new_block = False
        self.game_solved=False
        self.snake_body=[Vector2(3,int(self.cell_number/2)),Vector2(2,int(self.cell_number/2)),Vector2(1,int(self.cell_number/2))]
        self.direction = Vector2(1,0)
        self.empty_game_table=self.get_empty_game_table()
        self.update_game_table()
        self.randomize()
        self.update_game_table()
        

    def get_free_coords(self):
        free_coords=[]
        for x in range(self.cell_number):
            for y in range(self.cell_number):
                coord=Vector2(x,y)
                if self.game_table[x+self.map_margin,y+self.map_margin]==0:
                    free_coords.append(coord)
        return free_coords
        
    def randomize(self):
        free_coords=self.get_free_coords()
        if len(free_coords)>0:
            self.snack_pos=random.choice(free_coords)
        else:
            print('game solved!')
            self.game_solved=True
    
    def update_game_table(self):
        self.game_table=self.empty_game_table.copy()
        if hasattr(self, 'snack_pos'):
            self.game_table[int(self.snack_pos[0])+self.map_margin,int(self.snack_pos[1])+self.map_margin]=1
        for coords in self.snake_body:
            if self.game_table[int(coords[0])+self.map_margin,int(coords[1])+self.map_margin]!=2:#not overwriting hurdles and boundary
                self.game_table[int(coords[0])+self.map_margin,int(coords[1])+self.map_margin]=3#overwrites snack if head is on snack
    
        
    def get_empty_game_table(self):
        game_table=np.ones((self.cell_number+2*self.map_margin,self.cell_number+2*self.map_margin), dtype=np.uint8)*2#border
        game_table[self.map_margin:-self.map_margin,self.map_margin:-self.map_margin]*=0
        if self.random_hurdles:
            indices_x = np.random.choice(np.arange(game_table.shape[0]), replace=False,size=game_table.shape[0])
            indices_y = np.random.choice(np.arange(game_table.shape[1]), replace=False,size=game_table.shape[1])
            game_table[indices_x,indices_y] = 2
        return game_table
        
    
    def apply_action(self,action):
        if action==0:
            #left
            self.turn_left()
        elif action==1:
            #right
            self.turn_right()
        else:
            #do nothing
            pass
    
    def update(self,action):
        self.apply_action(action)
        self.move_snake()
        self.update_game_table()
        reward=self.get_reward()
        self.check_snack_found()
        return reward
    
    def get_reward(self):
        head_coords_x=int(self.snake_body[0][0])
        head_coords_y=int(self.snake_body[0][1])
        table_value=self.game_table[head_coords_x+self.map_margin,head_coords_y+self.map_margin]
        if self.snack_found():
            #snack found
            return 10
        elif self.is_fail():
            return -10
        else:
            return 0
        
    def snack_found(self):
        return self.snack_pos == self.snake_body[0]

    def check_snack_found(self):
        if self.snack_found():
            self.randomize()
            self.add_block()

    def is_fail(self):
        if not 0 <= self.snake_body[0].x < self.cell_number or not 0 <= self.snake_body[0].y < self.cell_number:
            return True
        
        for block in self.snake_body[1:]:
            if block == self.snake_body[0]:
                return True
        if self.game_table[int(self.snake_body[0].x+self.map_margin),int(self.snake_body[0].y+self.map_margin)]==2:
            return True
        return False
    


class Agent:
	#the agent/snake can observe the environment, decide for actions and receives reward. 
	#based on that it learns the expected reward of state-action pairs, i.e. the Q values
    def __init__(self,field_of_view_size=1,n_cell_states=3,gamma=0.9):
        self.gamma=gamma
        self.n_cell_states=n_cell_states
        self.field_of_view_size=field_of_view_size
        self.map_margin=1+self.field_of_view_size
        self.kernel_size=1+2*field_of_view_size
        self.n_kernel_elements=self.kernel_size**2
        self.n_possible_kernels=n_cell_states**(self.n_kernel_elements)
        self.Q=np.zeros((8*self.n_possible_kernels,3))#|D_snack|=8,|S|=n_cell_states**(D**2),|A|=3: l,r,straight
        self.n_visited=np.zeros((8*self.n_possible_kernels,3))#we keep track on how many times a state-action pair has been visited. This is a metric for the confidence
        
    def map_state_to_number_state(self,map_state):
        number_state=0
        count=0
        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                number_state+=map_state[x,y]*(self.n_cell_states**count)
                count+=1
        return int(number_state)
    
    def number_state_to_map_state(self,number_state):
        map_state=np.zeros((self.kernel_size,self.kernel_size))
        for i in range(self.n_kernel_elements):
            x=int(i/self.kernel_size)
            y=i-x*self.kernel_size
            map_state[x,y]=int(number_state%(self.n_cell_states**(i+1))/self.n_cell_states**i)
        return map_state
    
    def get_snack_number(self,snack_direction,head_direction):
        angle = snack_direction.angle_to(head_direction)
        angle_quantified=np.round(angle/360*8)
        angle_state=np.mod(angle_quantified,8)
        return angle_state


    def reflect_snack_direction(self,snack_direction,head_direction):
        reflected_snack_direction=Vector2((snack_direction[0],snack_direction[1]))
        reflected_snack_direction.reflect_ip(head_direction.rotate(90))
        return reflected_snack_direction
    
    def get_state(self,map_state,head_direction,snack_direction):
        reflected_snack_direction=self.reflect_snack_direction(snack_direction,head_direction)
        direction_number=self.direction_to_direction_state(head_direction)#assigns a number to the 4 possible direction
        for i in range(direction_number):
            map_state=np.rot90(map_state)#rotates the map equivariant to the direction
        reflected_map_state=np.flip(map_state,1)
        map_number=self.map_state_to_number_state(map_state)#assigns a unique number to each 1st person mapstate
        reflected_map_number=self.map_state_to_number_state(reflected_map_state)
        snack_number=self.get_snack_number(snack_direction,head_direction)
        reflected_snack_number=self.get_snack_number(reflected_snack_direction,head_direction)
        state=self.get_state_from_map_and_snack_number(map_number,snack_number)
        reflected_state=self.get_state_from_map_and_snack_number(reflected_map_number,reflected_snack_number)

        return state,reflected_state
        
        
    def observe_environment(self,environment):
        map_state=self.get_map_state(environment)
        head_direction=environment.direction
        head_position=environment.snake_body[0]
        snack_position=environment.snack_pos
        snack_direction=snack_position-head_position
        state,reflected_state=self.get_state(map_state,head_direction,snack_direction)
        return state,reflected_state
    
    def get_state_from_map_and_snack_number(self,map_number,snack_number):
        return int(map_number+snack_number*self.n_possible_kernels)
    
    def state_to_map_and_snack_number(self,state):
        snack_number=int(state/self.n_possible_kernels)
        map_number=state-snack_number*self.n_possible_kernels
        return map_number,snack_number
    
    def direction_to_direction_state(self,direction):
        if direction==Vector2(1,0):
            return 0
        elif direction==Vector2(0,-1):
            return 1
        elif direction==Vector2(-1,0):
            return 2
        elif direction==Vector2(0,1):
            return 3
        else:
            raise ValueError('Unknown direction: ',direction)
            
            
    def direction_state_to_direction(self,direction_state):
        if direction_state==0:
            return Vector2(1,0)
        elif direction_state==1:
            return Vector2(0,-1)
        elif direction_state==2:
            return Vector2(-1,0)
        elif direction_state==3:
            return Vector2(0,1)
        else:
            raise ValueError('Unknown direction_state: ',direction_state)
    
    def get_best_action_from_state(self,s):
        a_max=np.argmax(self.Q[s,:])
        Q_max=self.Q[s,a_max]
        return Q_max,a_max

    def update_Q(self,s_old,s_new,a,r,reward_only=False):
        Q_max,_=self.get_best_action_from_state(s_new)
        if reward_only:
            q=r
            self.Q[s_old,a]=r
        else:
            q=r+self.gamma*Q_max
        self.Q[s_old,a]=q#Bellman approximation
        self.n_visited[s_old,a]+=1
        return Q_max
    
    def get_map_state(self,environment):
        head_coords_x=int(environment.snake_body[0][0])
        head_coords_y=int(environment.snake_body[0][1])
        x_min=head_coords_x-self.field_of_view_size+self.map_margin
        y_min=head_coords_y-self.field_of_view_size+self.map_margin
        map_state=environment.game_table[x_min:x_min+self.kernel_size,y_min:y_min+self.kernel_size]
        if self.n_cell_states==3:
            map_state= np.clip(map_state,0,2)
        elif self.n_cell_states!=4:
            raise ValueError('n_cell_state ',self.n_cell_state,' is not implemented')
        return map_state
    
    def softmax(self,v,T=1):
        v_exp=np.exp(v/T)
        s=np.sum(v_exp)
        p=v_exp/s
        return p
    
    def get_action(self,observation,epsilon=0,T=1):
        min_action_visited=np.argmin(self.n_visited[observation,:])
        if self.n_visited[observation,min_action_visited]==0:
            a=min_action_visited
            p=np.zeros(3)
            p[a]=1
            #print('explored new state-action pair')
        else:
            if np.random.rand()<epsilon:
                p=[0.33,0.33,0.34]
            else:
                p=self.softmax(self.Q[observation,:],T)
            a=np.random.choice(3, p=p)
        if a==2:
        	reflected_a=a
        else:
            reflected_a=-a+1
        return a,reflected_a,p


def training_iter(agent,environment,graphics=None,with_sound=True,train_with_reflected_state=True,T=1):
    observation,reflected_observation=agent.observe_environment(environment)
    action,reflected_action,p=agent.get_action(observation,T=T)
    reward=environment.update(action)
    new_observation,_=agent.observe_environment(environment)
    agent.update_Q(observation,new_observation,action,reward,reward_only=environment.is_fail())
    if train_with_reflected_state:
    	agent.update_Q(reflected_observation,new_observation,reflected_action,reward,reward_only=environment.is_fail())
    if environment.snack_found() and with_sound:
        if graphics is not None:
            graphics.play_crunch_sound()
        else:
            print('Warning, cant play sound because graphics is not passed.')
    return reward

def close_pygame():
    pygame.quit()
    sys.exit()

def pretrain(n_epochs,environment,agent,train_with_reflected_state=True):
    game_rewards=[]
    normed_body_length=[]
    longest_body_length=0
    for epoch in range(n_epochs):
        environment.reset()
        old_observation=agent.observe_environment(environment)
        count=0
        count_snacks=0
        game_reward=0
        while not environment.is_fail():
            reward=training_iter(agent,environment,with_sound=False,train_with_reflected_state=train_with_reflected_state)
            game_reward+=reward
            count+=1
        game_rewards.append(game_reward/(count+1e-8))
        normed_body_length.append(len(environment.snake_body))
        if len(game_rewards)>2:
            game_rewards[-1]=0.98*game_rewards[-2]+0.02*game_rewards[-1]
            normed_body_length[-1]=0.98*normed_body_length[-2]+0.02*normed_body_length[-1]
            
        if (epoch+1)%50==0:
            print('epoch ',epoch,': ',game_rewards[-1])
            plt.xlabel('# games')
            plt.ylabel('total reward')
            plt.plot(game_rewards)
            plt.show()

            plt.xlabel('# games')
            plt.ylabel('body length')
            plt.plot(normed_body_length)
            plt.show()

        if len(environment.snake_body)>longest_body_length:
            longest_body_length=len(environment.snake_body)
            print('New longest bodylength: ',longest_body_length)


