#!/usr/bin/perl
#
#  Author: Sara Rosenthal, Preslav Nakov
#  
#  Description: Scores SemEval-2017 task 4, subtask A
#               Calculates macro-average R, macro-average F1, and Accuracy
#
#  Last modified: December 22, 2016
#
# Use:
# (a) outside of CodaLab
#     perl SemEval2016_task4_test_scorer_subtaskA.pl <GOLD_FILE> <INPUT_FILE>
# (b) with CodaLab, i.e., $codalab=1 (certain formatting is expected)
#     perl SemEval2016_task4_test_scorer_subtaskA.pl <INPUT_FILE> <OUTPUT_DIR>
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE   =  $ARGV[0];
my $INPUT_FILE  =  $ARGV[1];
my $OUTPUT_FILE =  $INPUT_FILE . '.scored';

my $codalab = 1; # set to 1 if the script is being used in CodaLab

########################
###   MAIN PROGRAM   ###
########################

my %stats = ();

### 1. Read the files and get the statistics
if ($codalab) {
	my $INPUT_DIR = $ARGV[0];
	print STDERR "Loading input from dir: $INPUT_DIR\n";
 
	opendir(DIR, "$INPUT_DIR/res/") or die $!;

	while (my $file = readdir(DIR)) {

	    # Use a regular expression to ignore files beginning with a period
    	    next if ($file =~ m/^(\.|_)/);
	    $INPUT_FILE = "$INPUT_DIR/res/$file";
	    last;
	}
	closedir(DIR);
	$GOLD_FILE   = "$INPUT_DIR/ref/twitter-2016test-A-English.txt";
	$OUTPUT_FILE = $ARGV[1] . "/scores.txt";
}

### 3. Calculate the F1, etc.
print STDERR "Found input file: $INPUT_FILE\n";

open INPUT, $INPUT_FILE or die;

print STDERR "Loading ref data $GOLD_FILE\n";
open GOLD,  $GOLD_FILE or die;

print STDERR "Loading the file...";
for (<INPUT>) {
	s/^[ \t]+//;
	s/[ \n\r]+$//;
	
	### 1.1. Check the input file format
	#1	positive	i'm done writing code for the week! Looks like we've developed a good a** game for the show Revenge on ABC Sunday, Premeres 09/30/12 9pm
	die "Wrong file format for $INPUT_FILE: '$_'" if (!/^(\d+)\t(positive|negative|neutral)/);
	my ($pid, $proposedLabel) = ($1, $2);

	### 1.2	. Check the gold file format
	#NA	T14114531	positive
	$_ = <GOLD>;
	die "Wrong file format!" if (!/^(\d+)\t(positive|negative|neutral)/);
	my ($tid, $trueLabel) = ($1, $2);

    	die "Ids mismatch!" if ($pid ne $tid);

	### 1.3. Update the statistics
	$stats{$proposedLabel}{$trueLabel}++;

}

while (<GOLD>) {
	die "Missing answer for the following tweet: '$_'\n";
}
print STDERR "DONE\n";

close(INPUT) or die;
close(GOLD) or die;

print STDERR "Calculating the scores...\n";

### 2. Initialize zero counts
foreach my $class1 ('positive', 'negative', 'neutral') {
	foreach my $class2 ('positive', 'negative', 'neutral') {
		$stats{$class1}{$class2} = 0 if (!defined($stats{$class1}{$class2}));
	}
}

### 3. Calculate the F1 for each dataset
print STDERR "Creating output file: $OUTPUT_FILE\n";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;
my $avgR  = 0.0;
my $avgF1 = 0.0;
foreach my $class ('positive', 'negative', 'neutral') {

	my $denomP = (($stats{$class}{'positive'} + $stats{$class}{'negative'} + $stats{$class}{'neutral'}) > 0) ? ($stats{$class}{'positive'} + $stats{$class}{'negative'} + $stats{$class}{'neutral'}) : 1;
	my $P = $stats{$class}{$class} / $denomP;

	my $denomR = ($stats{'positive'}{$class} + $stats{'negative'}{$class} + $stats{'neutral'}{$class}) > 0 ? ($stats{'positive'}{$class} + $stats{'negative'}{$class} + $stats{'neutral'}{$class}) : 1;
	my $R = $stats{$class}{$class} / $denomR;
			
	my $denom = ($P+$R > 0) ? ($P+$R) : 1;
	my $F1 = 2*$P*$R / $denom;

    $avgR  += $R;
	$avgF1 += $F1 if ($class ne 'neutral');
	printf OUTPUT "\t%8s: P=%0.4f, R=%0.4f, F1=%0.4f\n", $class, $P, $R, $F1 if (!$codalab);
}

$avgR  /= 3.0;
$avgF1 /= 2.0;
my $acc = ($stats{'positive'}{'positive'} + $stats{'negative'}{'negative'} + $stats{'neutral'}{'neutral'}) 
        / ($stats{'positive'}{'positive'} + $stats{'negative'}{'negative'} + $stats{'neutral'}{'neutral'} +
           $stats{'positive'}{'negative'} + $stats{'negative'}{'positive'} +
           $stats{'positive'}{'neutral'} + $stats{'neutral'}{'positive'} +
           $stats{'negative'}{'neutral'} + $stats{'neutral'}{'negative'});

if ($codalab) {
	printf OUTPUT "AverageF1: %0.3f\nAverageR: %0.3f\nAccuracy: %0.3f\n", $avgF1, $avgR, $acc;
}
else {
	printf OUTPUT "\t\tAvgF1_2=%0.3f, AvgR_3=%0.3f, Acc=%0.3f\n", $avgF1, $avgR, $acc;
	printf OUTPUT "\tOVERALL SCORE : %0.3f\n", $avgF1;
	print "$INPUT_FILE\t";
	printf "%0.3f\t%0.3f\t%0.3f\n", $avgF1, $avgR, $acc;
}
printf STDERR "AverageF1: %0.3f\nAverageR: %0.3f\nAccuracy: %0.3f\n", $avgF1, $avgR, $acc;
print STDERR "DONE\n";
print STDERR "Wrote to $OUTPUT_FILE\n";

close(OUTPUT) or die;
