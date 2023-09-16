{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.trials;

{$mode ObjFPC}{$H+}

interface

type

  TTrialParameters = record
    Name : string;
    Kind : string;
    Cursor : integer;
    RepeatTrials : integer;
    LimitedHold : integer;
    InterTrialInterval : integer;
    HasConsequence : Boolean;
  end;

  TTrialKeys = record
    Name : string;
    Cursor : string;
    Kind : string;
    LimitedHold : string;
    InterTrialInterval : string;
    RepeatTrials : string;
    HasConsequence : string;
  end;

const
  TrialKeys : TTrialKeys = (
    Name : 'Name';
    Cursor : 'Cursor';
    Kind : 'Kind';
    LimitedHold : 'LimitedHold';
    InterTrialInterval : 'InterTrialInterval';
    RepeatTrials : 'RepeatTrial';
    HasConsequence : 'HasConsequence');

implementation

end.

