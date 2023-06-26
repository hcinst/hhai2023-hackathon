# Git installation

For Windows.

```https://phoenixnap.com/kb/how-to-install-git-windows```


For MacOs

```brew install git```

For Debian based.

```sudo apt install git```

# Git configuration
``` 
git config --global user.name "your-username"
git config --global user.email "your-email"
```

# Cloning the project.

```
git clone https://github.com/hcinst/hhai-hackathon.git
```

# Project Installation

```
pip install -r requirements.txt
```

# Project Description

**collapse.py** - our original code for testing the successive collapsing method</br>
**megatiles.py** - cleaned up and commented code with insertion points for hackathon code (search for “HACKATHON” in multiple places)</br>
**pilot_megatiles_experts.csv** - annotations from two professional neuropathologists</br>
**pilot_megatiles_players.csv** - annotations from 30 non-expert volunteers</br>
**megatiles_stepwise_collapsing.txt** - pseudocode explaining our stepwise collapsing methods</br>

</br>

# Working on the project.

Run the following command to create a new branch and checkout on the branch.

```
git branch <your-name>-hackathon

git checkout <your-name>-hackathon
```

# Add your algorithm.

Search for “HACKATHON” in multiple places in megatiles.py and add your algorithm there.

# After adding your changes.
 Commit your changes to the branch.
 and push to github.

 ```
 git commit -am "your message"
 git push
 ```
<br>
SLIDES: https://docs.google.com/presentation/d/1_fHSHrW9Blkx22NCbTP89_3nllzQXmqeuOX0cPc3jD8/edit?usp=sharing
