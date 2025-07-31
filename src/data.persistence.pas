{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit data.persistence;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, IniPropStorage, PropertyStorage;

type

  { TDataPersistence }

  TDataPersistence = class(TDataModule)
    IniPropStorageProtocol: TIniPropStorage;
    procedure IniPropStorageProtocolStoredValues0Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorageProtocolStoredValues0Save(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorageProtocolStoredValues1Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorageProtocolStoredValues1Save(Sender: TStoredValue;
      var Value: TStoredType);
  private

  public

  end;

var
  DataPersistence: TDataPersistence;

implementation

{$R *.lfm}

uses Forms.Main;

{ TDataPersistence }

procedure TDataPersistence.IniPropStorageProtocolStoredValues0Restore(
  Sender: TStoredValue; var Value: TStoredType);
begin
  FormBackground.IniPropStorage1StoredValues0Restore(Sender, Value);
end;

procedure TDataPersistence.IniPropStorageProtocolStoredValues0Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  FormBackground.IniPropStorage1StoredValues0Save(Sender, Value);
end;

procedure TDataPersistence.IniPropStorageProtocolStoredValues1Restore(
  Sender: TStoredValue; var Value: TStoredType);
begin
  FormBackground.IniPropStorage1StoredValues1Restore(Sender, Value);
end;

procedure TDataPersistence.IniPropStorageProtocolStoredValues1Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  FormBackground.IniPropStorage1StoredValues1Save(Sender, Value);
end;

end.

